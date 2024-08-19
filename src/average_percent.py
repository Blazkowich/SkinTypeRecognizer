def average_percentages(data, duration=5, interval=1):
    """Collect data for a specified duration and compute average percentages."""
    start_time = time.time()
    combined_skin_list = []
    normal_skin_list = []
    dry_skin_list = []
    oily_skin_list = []

    while time.time() - start_time < duration:
        if 'combined' in data:
            combined_skin_list.append(data['combined'])
        if 'normal' in data:
            normal_skin_list.append(data['normal'])
        if 'dry' in data:
            dry_skin_list.append(data['dry'])
        if 'oily' in data:
            oily_skin_list.append(data['oily'])

        time.sleep(interval)

    # Calculate average percentages
    avg_combined = sum(combined_skin_list) / len(combined_skin_list) if combined_skin_list else 0
    avg_normal = sum(normal_skin_list) / len(normal_skin_list) if normal_skin_list else 0
    avg_dry = sum(dry_skin_list) / len(dry_skin_list) if dry_skin_list else 0
    avg_oily = sum(oily_skin_list) / len(oily_skin_list) if oily_skin_list else 0

    total = avg_normal + avg_oily + avg_dry + avg_combined
    if total > 0:
        avg_normal = (avg_normal / total) * 100
        avg_oily = (avg_oily / total) * 100
        avg_dry = (avg_dry / total) * 100
        avg_combined = (avg_combined / total) * 100

    return avg_combined, avg_normal, avg_dry, avg_oily
