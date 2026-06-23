# c:/Users/Carlo/Desktop/Camera-Processor-ros2/.venv/Scripts/python.exe scripts/compare_hz.py
import re
import os

def parse_hz_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return {}

    with open(file_path, 'r') as f:
        content = f.read()

    # Split by topic sections
    sections = re.split(r'=== .* ===', content)
    topic_data = {}

    for section in sections:
        topic_match = re.search(r'Topic: (.*)', section)
        if not topic_match:
            continue
        
        topic_name = topic_match.group(1).strip()
        
        # Check for warnings (topic not published)
        if "WARNING" in section:
            # We still want to track it, but mark as 0 or None
            # If there are NO rate lines at all, it's 0.
            # But sometimes it starts with warning and then gets messages.
            pass

        # Find all average rate values
        rates = re.findall(r'average rate: ([\d.]+)', section)
        if rates:
            float_rates = [float(r) for r in rates]
            avg_rate = sum(float_rates) / len(float_rates)
            topic_data[topic_name] = avg_rate
        else:
            topic_data[topic_name] = 0.0

    return topic_data

def compare_results(gui_file, ngui_file, output_file):
    gui_data = parse_hz_file(gui_file)
    ngui_data = parse_hz_file(ngui_file)

    all_topics = sorted(list(set(gui_data.keys()) | set(ngui_data.keys())))

    with open(output_file, 'w') as f:
        f.write("COMPARISON OF ROS2 TOPIC RATES (HZ)\n")
        f.write("====================================\n")
        f.write(f"GUI File: {gui_file}\n")
        f.write(f"NoGUI File: {ngui_file}\n\n")
        
        f.write(f"{'Topic':<40} | {'GUI (Hz)':<10} | {'NoGUI (Hz)':<10} | {'Diff (Hz)':<10} | {'Diff (%)':<10} | {'Winner':<10}\n")
        f.write("-" * 105 + "\n")

        total_gui = 0
        total_ngui = 0
        topic_count = 0

        for topic in all_topics:
            gui_hz = gui_data.get(topic, 0.0)
            ngui_hz = ngui_data.get(topic, 0.0)
            
            # Skip if topic was not active in BOTH tests to avoid dragging down averages
            if gui_hz == 0 or ngui_hz == 0:
                # Still output to file but mark as skipped for total summary
                diff_hz = ngui_hz - gui_hz
                diff_pct = 0.0
                winner = "N/A"
                f.write(f"{topic:<40} | {gui_hz:<10.3f} | {ngui_hz:<10.3f} | {diff_hz:<10.3f} | {'N/A':<11} | {winner:<10}\n")
                continue

            diff_hz = ngui_hz - gui_hz
            diff_pct = (diff_hz / gui_hz) * 100

            winner = "NoGUI" if diff_hz > 0.01 else ("GUI" if diff_hz < -0.01 else "Equal")
            
            f.write(f"{topic:<40} | {gui_hz:<10.3f} | {ngui_hz:<10.3f} | {diff_hz:<10.3f} | {diff_pct:<10.1f}% | {winner:<10}\n")
            
            total_gui += gui_hz
            total_ngui += ngui_hz
            topic_count += 1

        f.write("-" * 105 + "\n")
        
        if topic_count > 0:
            avg_gui = total_gui / topic_count
            avg_ngui = total_ngui / topic_count
            total_diff_hz = avg_ngui - avg_gui
            total_diff_pct = (total_diff_hz / avg_gui * 100) if avg_gui > 0 else 0
            
            final_winner = "NoGUI" if total_diff_hz > 0 else "GUI"
            better_pct = abs(total_diff_pct)
            
            f.write(f"\nOVERALL SUMMARY:\n")
            f.write(f"Average GUI Hz: {avg_gui:.3f}\n")
            f.write(f"Average NoGUI Hz: {avg_ngui:.3f}\n")
            f.write(f"Overall improvement: {total_diff_hz:.3f} Hz ({total_diff_pct:.1f}%)\n")
            f.write(f"Conclusion: {final_winner} was {better_pct:.1f}% better on average ({abs(total_diff_hz):.3f} Hz more).\n")

if __name__ == "__main__":
    gui_path = r'scripts/results_hz_GUI_jet.txt'
    ngui_path = r'scripts/results_hz_NGUI_jet.txt'
    out_path = r'scripts/results_comparison_GUIvsNGUI_jet.txt'
    
    print(f"Comparing {gui_path} and {ngui_path}...")
    compare_results(gui_path, ngui_path, out_path)
    print(f"Results saved to {out_path}")
