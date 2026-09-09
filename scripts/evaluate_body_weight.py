"""Reproducible isolated evaluation; individual targets are saved only in tmp/."""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from body_weight import (BodyRegionCalibrator, WeightCalibrator, average_empty_baseline,
    find_metadata_file, find_region_json, median_feature_row, parse_body_metadata,
    parse_json_frame, parse_region_label, sample_user_features, weight_interval)


def locate_default_data_root(project_root):
    configured = os.environ.get('MATTRESS_DATA_DIR')
    if configured:
        path = Path(configured)
        return path.parent if path.name == '睡姿数据' else path
    return next((p for p in project_root.parent.glob('*data') if (p / 'readme').is_file()), None)


def box_iou(actual, predicted):
    a,b,c,d = actual
    e,f,g,h = predicted
    intersection = max(0,min(b,f)-max(a,e))*max(0,min(d,h)-max(c,g))
    union = (b-a)*(d-c)+(f-e)*(h-g)-intersection
    return intersection/union if union else 0.0


def evaluate_regions(root):
    path = find_region_json(root)
    if path is None:
        raise ValueError('未找到区域标注')
    records = json.loads(path.read_text(encoding='utf-8'))
    records = [r for r in records if parse_region_label(r.get('region'))]
    grouped = {}
    for record in records:
        key = (record['people_name'], int(record['action']))
        grouped.setdefault(key, []).append(record)
    validation_records = []
    excluded_hashes = set()
    for (user, _), sequence in grouped.items():
        if user not in {'SAI', 'dgs'}:
            continue
        validation_start = round(len(sequence) * 0.7)
        # Keep one frame between chronological train and validation blocks.
        # Excluding hashes also prevents repeated frames from crossing the split.
        excluded = sequence[max(0, validation_start - 1):]
        excluded_hashes.update(sha256(r['data'].encode()).hexdigest() for r in excluded)
        validation_records.extend(sequence[validation_start:])
    validation_hashes = {sha256(r['data'].encode()).hexdigest() for r in validation_records}
    model = BodyRegionCalibrator.from_dataset(root, excluded_frame_hashes=excluded_hashes)
    if not model.samples or not model.test_users:
        raise ValueError('缺少隔离训练/测试用户')
    buckets = {k:dict(parts=0,tolerance_hits=0,exact_hits=0,iou_hits=0,iou_sum=0.0)
               for k in ('known_validation','new_users')}
    train_hashes = {sha256(r['data'].encode()).hexdigest() for r in records
                    if r['people_name'] in model.train_users
                    and sha256(r['data'].encode()).hexdigest() not in excluded_hashes}
    overlap = 0
    for r in records:
        if r['people_name'] in model.test_users:
            group = 'new_users'
        elif sha256(r['data'].encode()).hexdigest() in validation_hashes:
            group = 'known_validation'
        else:
            continue
        overlap += int(sha256(r['data'].encode()).hexdigest() in train_hashes)
        predicted = model.predict(parse_json_frame(r['data']))
        expected = parse_region_label(r['region'])
        score = buckets[group]
        for i,target in enumerate(expected):
            score['parts'] += 1
            if i >= len(predicted):
                continue
            box = tuple(predicted[i][k] for k in ('startRow','endRow','startCol','endCol'))
            score['tolerance_hits'] += int(all(abs(x-y)<=t for x,y,t in zip(box,target,(3,4,4,4))))
            score['exact_hits'] += int(box == target)
            iou = box_iou(target,box)
            score['iou_hits'] += int(iou >= 0.5)
            score['iou_sum'] += iou
    for score in buckets.values():
        n = score['parts']
        score.update(boundary_tolerance_accuracy_pct=round(100*score['tolerance_hits']/n,2) if n else None,
                     exact_box_accuracy_pct=round(100*score['exact_hits']/n,2) if n else None,
                     iou_0_5_accuracy_pct=round(100*score['iou_hits']/n,2) if n else None,
                     mean_iou=round(score['iou_sum']/n,4) if n else None)
        del score['iou_sum']
    return dict(seed=42,train_users=model.train_users,test_users=model.test_users,
                validation_strategy='chronological 70/30 blocks per known user and action, with a one-frame guard gap',
                validation_samples=len(validation_records),train_samples=len(model.samples),
                train_evaluation_duplicate_frames=overlap,labelled_users=3,
                criterion='primary: boundary tolerance row start/end 3/4, col start/end 4/4 cells; IoU >= 0.5 also reported',**buckets)


def summarize_weight(rows):
    n = len(rows)
    if not n:
        raise ValueError('没有可评估的样本，不能报告零误差')
    errors = [abs(r['predicted_kg']-r['actual_kg']) for r in rows]
    diffs = [abs(weight_interval(r['predicted_kg'])['index']-weight_interval(r['actual_kg'])['index']) for r in rows]
    return dict(count=n,mae_kg=round(sum(errors)/n,3),
        interval_hit_rate_pct=round(100*sum(d==0 for d in diffs)/n,2),
        interval_or_adjacent_hit_rate_pct=round(100*sum(d<=1 for d in diffs)/n,2),
        interval_or_adjacent_3kg_hit_rate_pct=round(100*sum(d==0 or (d==1 and e<=3) for d,e in zip(diffs,errors))/n,2),
        within_3kg_rate_pct=round(100*sum(e<=3 for e in errors)/n,2),
        far_miss_rate_pct=round(100*sum(d>=2 for d in diffs)/n,2))


def evaluate_weight(root):
    data_dir = root / '睡姿数据'
    if not data_dir.is_dir():
        raise ValueError('数据根目录必须包含睡姿数据')
    metadata = parse_body_metadata(find_metadata_file(data_dir))
    model = WeightCalibrator.from_dataset(data_dir)
    raw_model = WeightCalibrator.from_dataset(data_dir,correct_baseline=False)
    if not model.metadata_count or not model.test_users:
        raise ValueError('体重模型缺少隔离训练/测试用户')
    groups = {k:[] for k in ('known_users','new_users','new_user_single_frames','new_users_without_baseline')}
    baseline_count,empty_load = 0,[]
    for user in model.train_users+model.test_users:
        features = sample_user_features(data_dir/user)
        if not features:
            raise ValueError(f'用户无有效采集：{user}')
        baseline = average_empty_baseline(data_dir/user)
        prediction = model.predict_features(median_feature_row(features),user,baseline is not None)
        group = 'known_users' if user in model.train_users else 'new_users'
        groups[group].append(dict(user=user,actual_kg=metadata[user]['weight'],predicted_kg=prediction['kg']))
        baseline_count += int(baseline is not None)
        if baseline is not None:
            empty_load.append(sum(map(sum,baseline)))
        if group == 'new_users':
            raw = sample_user_features(data_dir/user,correct_baseline=False)
            pred = raw_model.predict_features(median_feature_row(raw),user,False)
            groups['new_users_without_baseline'].append(dict(user=user,actual_kg=metadata[user]['weight'],predicted_kg=pred['kg']))
            for feature in features:
                single = model.predict_features(feature,user,baseline is not None,'single_frame')
                if single['kg'] is not None:
                    groups['new_user_single_frames'].append(dict(user=user,actual_kg=metadata[user]['weight'],predicted_kg=single['kg']))
    summary = {k:summarize_weight(rows) for k,rows in groups.items()}
    summary.update(train_users=model.train_users,test_users=model.test_users,seed=42,
        sampling='first 3 frames from every available static sequence 1..21; per-user feature median',
        known_users_note='training fit only, not independent validation',
        height_input='measured height available to known and new users; no target weight used during prediction',
        baseline=dict(available_users=baseline_count,mean_total_empty_pressure=round(sum(empty_load)/len(empty_load),2) if empty_load else None),
        test_bin_counts=dict(Counter(str(weight_interval(metadata[u]['weight'])['index']) for u in model.test_users)))
    return summary,[dict(group=k,**r) for k,rows in groups.items() for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--output',type=Path,default=Path('results/body_weight_results.json'))
    parser.add_argument('--private-output',type=Path,default=Path('tmp/body_weight_predictions.json'))
    args = parser.parse_args()
    root = args.data_root or locate_default_data_root(Path(__file__).resolve().parents[1])
    if root is None or not root.is_dir():
        parser.error('请指定包含 readme、睡姿数据、区域划分的 --data-root')
    weight,private = evaluate_weight(root)
    region = evaluate_regions(root)
    result = dict(schema_version=2,region=region,weight=weight,acceptance=dict(
        region_known_gt95=region['known_validation']['boundary_tolerance_accuracy_pct']>95,
        region_new_gt70=region['new_users']['boundary_tolerance_accuracy_pct']>70,
        no_region_duplicate_leakage=region['train_evaluation_duplicate_frames']==0,
        weight_mae_le5=weight['new_users']['mae_kg']<=5,
        weight_interval_or_adjacent_ge85=weight['new_users']['interval_or_adjacent_hit_rate_pct']>=85,
        weight_far_miss_lt3=weight['new_users']['far_miss_rate_pct']<3))
    for path,content in ((args.output,result),(args.private_output,private)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
