# --- START OF FILE basis_path_report.py ---

#!/usr/bin/env python3
"""Generate basis path testing report for critical functions."""

import os
import subprocess
from radon.complexity import cc_visit
from coverage import Coverage
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Daftar fungsi yang ditargetkan tetap sama seperti sebelumnya
TARGET_FUNCTIONS = {
    # --- Modul Inti & Orkestrasi ---
    "main.py": [
        "main", 
        "analyze_image_comprehensive_advanced"
    ],
    "app2.py": [
        "ForensicValidator.validate_cross_algorithm",
        "lakukan_validasi_sistem"
    ],
    "utils.py": [
        "delete_selected_history", 
        "load_analysis_history"
    ],
    # --- Modul Analisis Lanjutan ---
    "advanced_analysis.py": [
        "analyze_noise_consistency",
        "perform_statistical_analysis",
        "analyze_frequency_domain",
        "analyze_texture_consistency",
        "analyze_edge_consistency",
        "analyze_illumination_consistency"
    ],
    # --- Modul Deteksi & Klasifikasi ---
    "classification.py": [
        "classify_manipulation_advanced", 
        "prepare_feature_vector"
    ],
    "copy_move_detection.py": [
        "detect_copy_move_blocks",
        "detect_copy_move_advanced",
        "kmeans_tampering_localization"
    ],
    "feature_detection.py": ["extract_multi_detector_features"],
    # --- Modul Analisis Spesifik ---
    "ela_analysis.py": ["perform_multi_quality_ela"],
    "jpeg_analysis.py": [
        "comprehensive_jpeg_analysis",
        "detect_double_jpeg",
        "jpeg_ghost_analysis",
        "analyze_jpeg_blocks",
    ],
    # --- Modul Validasi & Ekspor ---
    "validation.py": [
        "validate_image_file", 
        "extract_enhanced_metadata",
        "advanced_preprocess_image"
    ],
    "visualization.py": [
        "visualize_results_advanced",
        "create_advanced_combined_heatmap",
    ],
    "export_utils.py": [
        "export_to_advanced_docx",
        "generate_all_process_images"
    ],
}


def analyze_complexity():
    """Compute cyclomatic complexity and line numbers for targeted functions."""
    complexity_data = {}
    line_map = {}
    for filename, funcs in TARGET_FUNCTIONS.items():
        if not os.path.exists(filename):
            print(f"Warning: File '{filename}' not found, skipping analysis.")
            continue
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                code = f.read()
            for block in cc_visit(code):
                if block.name in funcs:
                    full_name = f"{os.path.splitext(filename)[0].replace(os.sep, '.')}.{block.name}"
                    complexity_data[full_name] = block.complexity
                    line_map[full_name] = (filename, block.lineno, block.endline)
        except Exception as e:
            print(f"Could not analyze complexity for {filename}: {e}")
    return complexity_data, line_map


def run_tests_and_get_bugs():
    """Run pytest with coverage and get bug count."""
    cmd = ["pytest", "--cov=."]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    output = proc.stdout + proc.stderr
    bug_count = 0
    if "failed" in output:
        for line in output.splitlines():
            if "failed" in line and "passed" in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "failed":
                            bug_count = int(parts[i - 1])
                            break
                except (ValueError, IndexError):
                    continue
    return bug_count


def calculate_function_coverage(cov, line_map):
    """Calculate coverage for each specific function using its line numbers."""
    coverage_info = {}
    for func_name, (filename, start_line, end_line) in line_map.items():
        try:
            analysis = cov.analysis2(filename)
            executed_lines = set(analysis[1])
            function_lines = set(range(start_line, end_line + 1))
            if not function_lines:
                coverage_info[func_name] = 0.0
                continue
            covered_lines_in_function = function_lines.intersection(executed_lines)
            coverage_percent = (len(covered_lines_in_function) / len(function_lines)) * 100
            coverage_info[func_name] = coverage_percent
        except Exception:
            coverage_info[func_name] = 0.0
    return coverage_info

# ======================= FUNGSI PEMBUATAN LAPORAN BARU =======================

def generate_markdown_report(report_data):
    """Generates a well-formatted Markdown report."""
    headers = ["Fungsi", "Cyclomatic Complexity", "Jalur Dasar", "Cakupan Jalur (%)", "Bug Ditemukan"]
    
    # Hitung lebar kolom untuk perataan
    col_widths = {h: len(h) for h in headers}
    for item in report_data:
        col_widths["Fungsi"] = max(col_widths["Fungsi"], len(item['name']))
        col_widths["Cyclomatic Complexity"] = max(col_widths["Cyclomatic Complexity"], len(str(item['complexity'])))
        col_widths["Jalur Dasar"] = max(col_widths["Jalur Dasar"], len(str(item['complexity'])))
        col_widths["Cakupan Jalur (%)"] = max(col_widths["Cakupan Jalur (%)"], len(f"{item['coverage']:.1f}"))
        col_widths["Bug Ditemukan"] = max(col_widths["Bug Ditemukan"], len(str(item['bugs'])))

    # Buat header tabel
    header_line = "| " + " | ".join([h.ljust(col_widths[h]) for h in headers]) + " |"
    separator_line = "|-" + "-|-".join(["-" * col_widths[h] for h in headers]) + "-|"
    
    report_lines = ["# Hasil Basis Path Testing\n", header_line, separator_line]

    # Buat baris data
    for item in report_data:
        row = [
            f"`{item['name']}`".ljust(col_widths["Fungsi"]),
            str(item['complexity']).ljust(col_widths["Cyclomatic Complexity"]),
            str(item['complexity']).ljust(col_widths["Jalur Dasar"]),
            f"{item['coverage']:.1f}".ljust(col_widths["Cakupan Jalur (%)"]),
            str(item['bugs']).ljust(col_widths["Bug Ditemukan"]),
        ]
        report_lines.append("| " + " | ".join(row) + " |")

    try:
        with open("BASIS_PATH_REPORT.md", "w", encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print("✅ Laporan Markdown 'BASIS_PATH_REPORT.md' berhasil dibuat.")
    except Exception as e:
        print(f"❌ Gagal menulis laporan Markdown: {e}")

def generate_xml_report(report_data, bug_count):
    """Generates an XML report."""
    root = ET.Element("BasisPathTestingReport")
    
    summary = ET.SubElement(root, "Summary")
    ET.SubElement(summary, "GeneratedDate").text = datetime.now().isoformat()
    ET.SubElement(summary, "BugsFound").text = str(bug_count)

    functions = ET.SubElement(root, "Functions")
    
    for item in report_data:
        func_elem = ET.SubElement(functions, "Function")
        ET.SubElement(func_elem, "Name").text = item['name']
        ET.SubElement(func_elem, "CyclomaticComplexity").text = str(item['complexity'])
        ET.SubElement(func_elem, "BasePathCount").text = str(item['complexity'])
        ET.SubElement(func_elem, "PathCoveragePercentage").text = f"{item['coverage']:.1f}"

    # Pretty print XML
    xml_str = ET.tostring(root, 'utf-8')
    parsed_str = minidom.parseString(xml_str)
    pretty_xml_str = parsed_str.toprettyxml(indent="  ")
    
    try:
        with open("BASIS_PATH_REPORT.xml", "w", encoding='utf-8') as f:
            f.write(pretty_xml_str)
        print("✅ Laporan XML 'BASIS_PATH_REPORT.xml' berhasil dibuat.")
    except Exception as e:
        print(f"❌ Gagal menulis laporan XML: {e}")


def generate_html_report(report_data, bug_count):
    """Generates an HTML report."""
    # Fungsi helper untuk pewarnaan sel coverage
    def get_coverage_color(coverage):
        if coverage >= 80:
            return "#d4edda"  # Green
        elif coverage >= 50:
            return "#fff3cd"  # Yellow
        else:
            return "#f8d7da"  # Red
            
    html_template = f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <title>Laporan Basis Path Testing</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 2em; background-color: #f4f4f9; }}
            h1 {{ color: #333; }}
            p {{ color: #555; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); background-color: #fff; }}
            th, td {{ padding: 12px 15px; border: 1px solid #ddd; text-align: left; }}
            thead tr {{ background-color: #4CAF50; color: white; }}
            tbody tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tbody tr:hover {{ background-color: #e2e2e2; }}
            code {{ background-color: #eee; padding: 2px 5px; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <h1>Laporan Hasil Basis Path Testing</h1>
        <p><strong>Tanggal Dibuat:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Total Bug Ditemukan dari Test Suite:</strong> {bug_count}</p>
        <table>
            <thead>
                <tr>
                    <th>Fungsi</th>
                    <th>Cyclomatic Complexity</th>
                    <th>Jalur Dasar</th>
                    <th>Cakupan Jalur (%)</th>
                </tr>
            </thead>
            <tbody>
    """

    for item in report_data:
        html_template += f"""
                <tr>
                    <td><code>{item['name']}</code></td>
                    <td>{item['complexity']}</td>
                    <td>{item['complexity']}</td>
                    <td style="background-color: {get_coverage_color(item['coverage'])};">{item['coverage']:.1f}</td>
                </tr>
        """
    
    html_template += """
            </tbody>
        </table>
    </body>
    </html>
    """

    try:
        with open("BASIS_PATH_REPORT.html", "w", encoding='utf-8') as f:
            f.write(html_template)
        print("✅ Laporan HTML 'BASIS_PATH_REPORT.html' berhasil dibuat.")
    except Exception as e:
        print(f"❌ Gagal menulis laporan HTML: {e}")

# ======================= FUNGSI ORKESTRATOR UTAMA =======================
def generate_report():
    print("Step 1: Analyzing code complexity...")
    complexity, line_map = analyze_complexity()
    if not complexity:
        print("Fatal error: No target functions found or files could not be read. Aborting.")
        return

    print("Step 2: Running tests to generate coverage data and count bugs...")
    bug_count = run_tests_and_get_bugs()

    print("Step 3: Calculating function-specific coverage...")
    try:
        cov = Coverage()
        cov.load()
        coverage_data = calculate_function_coverage(cov, line_map)
    except Exception as e:
        print(f"Fatal error processing coverage data: {e}")
        coverage_data = {func: 0.0 for func in complexity}

    # Kumpulkan semua data ke dalam satu struktur
    report_data = []
    sorted_functions = sorted(complexity.items(), key=lambda item: item[0])
    for func_name, comp in sorted_functions:
        report_data.append({
            'name': func_name,
            'complexity': comp,
            'coverage': coverage_data.get(func_name, 0.0),
            'bugs': bug_count # Bug count is global for the suite
        })

    print("\nStep 4: Generating reports in multiple formats...")
    generate_markdown_report(report_data)
    generate_xml_report(report_data, bug_count)
    generate_html_report(report_data, bug_count)
    print("\nAll reports generated successfully!")


if __name__ == "__main__":
    generate_report()
# --- END OF FILE basis_path_report.py ---