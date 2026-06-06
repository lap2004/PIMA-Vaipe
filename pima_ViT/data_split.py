import json
import random
import os

def load_data(map_path):
    with open(map_path, 'r') as f:
        data = json.load(f)
    return data

def greedy_split(data, target_pres_train_ratio, target_pill_train_ratio):
    total_pres = len(data)
    total_pills = sum(len(item['pill']) for item in data)
    
    target_pres_train = int(total_pres * target_pres_train_ratio)
    target_pill_train = int(total_pills * target_pill_train_ratio)
    
    # Sort data by number of pills (descending) to pack the largest first
    sorted_data = sorted(data, key=lambda x: len(x['pill']), reverse=True)
    
    train_data = []
    test_data = []
    
    current_pres_train = 0
    current_pill_train = 0
    
    # Randomize a bit to not always get the same split for same ratios, but keep sorted order roughly
    # Actually, let's just do a greedy approach with a bit of randomness
    random.shuffle(data)
    
    for item in data:
        num_pills = len(item['pill'])
        # Check if adding this item brings us closer to the target or pushes us away
        if current_pres_train < target_pres_train and current_pill_train < target_pill_train:
            train_data.append(item)
            current_pres_train += 1
            current_pill_train += num_pills
        else:
            test_data.append(item)
            
    # Calculate actual percentages
    actual_pres_train_ratio = len(train_data) / total_pres if total_pres > 0 else 0
    actual_pill_train_ratio = current_pill_train / total_pills if total_pills > 0 else 0
    
    print(f"Target Pres Train: {target_pres_train_ratio:.4f}, Actual: {actual_pres_train_ratio:.4f}")
    print(f"Target Pill Train: {target_pill_train_ratio:.4f}, Actual: {actual_pill_train_ratio:.4f}")
    
    return train_data, test_data

def main():
    map_path = '/home/lab/son/lap/vaipepill2022/public_train/pill_pres_map.json'
    out_dir = '/home/lab/son/lap/PIMA_NEW/splits'
    os.makedirs(out_dir, exist_ok=True)
    
    data = load_data(map_path)
    
    scenarios = {
        '1-1': (0.6955, 0.7058),
        '1-2': (0.3889, 0.4504),
        '1-3': (0.0386, 0.0612),
        '2-1_2-2': (0.6955, 0.7240)
    }
    
    for scenario, (target_pres, target_pill) in scenarios.items():
        print(f"\n--- Scenario {scenario} ---")
        train_data, test_data = greedy_split(data, target_pres, target_pill)
        
        with open(os.path.join(out_dir, f'train_scenario_{scenario}.json'), 'w') as f:
            json.dump(train_data, f, indent=4)
        with open(os.path.join(out_dir, f'test_scenario_{scenario}.json'), 'w') as f:
            json.dump(test_data, f, indent=4)

if __name__ == "__main__":
    main()
