#!/usr/bin/env python
"""
Analyze feedback logs to generate a summary report

This script reads the feedback logs generated during validation and regeneration
cycles, and creates a comprehensive report showing how errors were reduced
and code quality improved across iterations.
"""

import os
import sys
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Ensure proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_logs_by_run(logs_dir="feedback_logs"):
    """Group log files by run_id and sort by iteration"""
    logs_by_run = defaultdict(list)
    
    # Find all log files
    log_files = glob.glob(os.path.join(logs_dir, "*.json"))
    
    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                log_data = json.load(f)
                
                # Extract key information
                run_type = log_data.get("run_type", "unknown")
                run_id = log_data.get("run_id", 0)
                iteration = log_data.get("iteration", 0)
                
                # Create a unique key for this run
                run_key = f"{run_type}_{run_id}"
                
                # Store the log data with its file path
                logs_by_run[run_key].append({
                    "file_path": log_file,
                    "data": log_data,
                    "iteration": iteration
                })
        except Exception as e:
            print(f"Error processing {log_file}: {str(e)}")
    
    # Sort each run's logs by iteration
    for run_key in logs_by_run:
        logs_by_run[run_key].sort(key=lambda x: x["iteration"])
    
    return logs_by_run

def count_errors(validation_feedback):
    """Count approximate number of errors in feedback"""
    if not validation_feedback:
        return 0
        
    # Split by sections (## Terraform Syntax Issues, ## Compliance Issues)
    sections = validation_feedback.split("##")
    
    # Count numbered items (lines starting with digit + period)
    error_count = 0
    for section in sections:
        lines = section.split("\n")
        for line in lines:
            if line.strip() and line.strip()[0].isdigit() and ". " in line:
                error_count += 1
    
    return error_count

def count_detailed_errors(validation_results):
    """Count different types of errors from validation results directly"""
    if not validation_results:
        return {
            "syntax_errors": 0,
            "compliance_errors": 0,
            "total_errors": 0
        }
    
    # Count syntax errors
    syntax_errors = len(validation_results.get("errors", []))
    
    # Count compliance errors (Checkov failures)
    compliance_errors = 0
    checkov_results = validation_results.get("checkov_results", {})
    if checkov_results and "results" in checkov_results:
        failed_checks = checkov_results.get("results", {}).get("failed_checks", [])
        compliance_errors = len(failed_checks)
    
    return {
        "syntax_errors": syntax_errors,
        "compliance_errors": compliance_errors,
        "total_errors": syntax_errors + compliance_errors
    }

def analyze_error_progression(logs):
    """Analyze how errors progress through iterations"""
    results = []
    
    for run_key, run_logs in logs.items():
        # Fix the splitting to handle run types that contain underscores
        # The run_key format is typically "run_type_run_id" where run_type might contain underscores
        parts = run_key.split("_")
        # The last part is the run_id, everything before is the run_type
        run_id = parts[-1]
        run_type = "_".join(parts[:-1])
        
        # Extract the question from the first log
        if run_logs:
            question = run_logs[0]["data"].get("question", "Unknown query")
        else:
            continue
            
        # Track errors across iterations
        for log_entry in run_logs:
            data = log_entry["data"]
            iteration = data.get("iteration", 0)
            
            # Extract validation results
            validation_results = data.get("validation_results", {})
            validation_feedback = data.get("validation_feedback", "")
            
            # Count errors using both methods
            feedback_error_count = count_errors(validation_feedback)
            detailed_errors = count_detailed_errors(validation_results)
            
            # Status flags
            syntax_valid = validation_results.get("syntax_valid", False)
            cis_compliant = validation_results.get("cis_compliant", False)
            
            # Store results
            results.append({
                "run_type": run_type,
                "run_id": run_id,
                "question": question,
                "iteration": iteration,
                "syntax_valid": syntax_valid,
                "cis_compliant": cis_compliant,
                "feedback_error_count": feedback_error_count,
                "syntax_errors": detailed_errors["syntax_errors"],
                "compliance_errors": detailed_errors["compliance_errors"],
                "total_errors": detailed_errors["total_errors"]
            })
    
    # Convert to DataFrame
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=[
            "run_type", "run_id", "question", "iteration", 
            "syntax_valid", "cis_compliant", 
            "feedback_error_count", "syntax_errors", "compliance_errors", "total_errors"
        ])

def generate_error_charts(df, output_dir="feedback_reports"):
    """Generate charts showing error reduction"""
    if df.empty:
        print("No data to generate charts")
        return
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by run_type
    for run_type, group in df.groupby("run_type"):
        # Create a chart for each run_id in this run_type
        for run_id, run_group in group.groupby("run_id"):
            # Sort by iteration
            run_group = run_group.sort_values("iteration")
            
            if len(run_group) <= 1:
                continue  # Skip runs with only one iteration
                
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot error counts
            plt.plot(run_group["iteration"], run_group["total_errors"], 
                     marker='o', linewidth=2, label="Total Errors")
            plt.plot(run_group["iteration"], run_group["syntax_errors"], 
                     marker='s', linewidth=2, label="Syntax Errors")
            plt.plot(run_group["iteration"], run_group["compliance_errors"], 
                     marker='^', linewidth=2, label="Compliance Errors")
            
            # Ensure y-axis starts at 0 and has reasonable scale
            max_errors = run_group[["total_errors", "syntax_errors", "compliance_errors"]].max().max()
            plt.ylim(bottom=0, top=max(max_errors + 1, 5))  # Ensure reasonable height
            
            # Add validation status
            for i, row in run_group.iterrows():
                status_text = []
                if row["syntax_valid"]:
                    status_text.append("Valid Syntax")
                if row["cis_compliant"]:
                    status_text.append("CIS OK")
                
                if status_text:
                    plt.annotate(
                        " + ".join(status_text), 
                        (row["iteration"], 0.5),  # Position at bottom of chart
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green", alpha=0.7)
                    )
            
            # Add details
            question = run_group["question"].iloc[0]
            if len(question) > 50:
                question = question[:47] + "..."
                
            plt.title(f"Error Reduction - {question}", fontsize=14)
            plt.xlabel("Iteration", fontsize=12)
            plt.ylabel("Error Count", fontsize=12)
            plt.xticks(run_group["iteration"])
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='upper right')
            
            # Add error count annotations
            for i, row in run_group.iterrows():
                if row["total_errors"] > 0:
                    plt.annotate(
                        f"{row['total_errors']}",
                        (row["iteration"], row["total_errors"]),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center'
                    )
            
            # Adjust layout and save chart
            plt.tight_layout()
            chart_file = os.path.join(output_dir, f"{run_type}_{run_id}_errors.png")
            plt.savefig(chart_file, dpi=120)
            plt.close()
            
            print(f"Generated chart: {chart_file}")

def generate_summary_report(df, output_dir="feedback_reports"):
    """Generate a summary report of feedback analysis"""
    if df.empty:
        print("No data to generate summary report")
        return
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare report data
    summary_rows = []
    
    # Group by run and analyze improvement
    for (run_type, run_id), group in df.groupby(["run_type", "run_id"]):
        # Sort by iteration
        group = group.sort_values("iteration")
        
        # Skip runs with only one iteration
        if len(group) <= 1:
            continue
            
        # Get first and last iteration
        first = group.iloc[0]
        last = group.iloc[-1]
        
        # Calculate improvement
        error_reduction = first["total_errors"] - last["total_errors"]
        error_reduction_pct = (error_reduction / first["total_errors"] * 100) if first["total_errors"] > 0 else 0
        
        # Calculate compliance improvement
        compliance_improvement = first["compliance_errors"] - last["compliance_errors"]
        compliance_pct = (compliance_improvement / first["compliance_errors"] * 100) if first["compliance_errors"] > 0 else 0
        
        summary_rows.append({
            "run_type": run_type,
            "query": first["question"],
            "iterations": len(group),
            "initial_errors": first["total_errors"],
            "final_errors": last["total_errors"],
            "error_reduction": error_reduction,
            "error_reduction_pct": error_reduction_pct,
            "initial_compliance_issues": first["compliance_errors"],
            "final_compliance_issues": last["compliance_errors"],
            "compliance_improvement": compliance_improvement,
            "compliance_improvement_pct": compliance_pct,
            "final_syntax_valid": last["syntax_valid"],
            "final_cis_compliant": last["cis_compliant"]
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # Generate HTML report
    if not summary_df.empty:
        # Format the DataFrame for HTML
        formatted_df = summary_df.copy()
        formatted_df["error_reduction_pct"] = formatted_df["error_reduction_pct"].map("{:.1f}%".format)
        formatted_df["compliance_improvement_pct"] = formatted_df["compliance_improvement_pct"].map("{:.1f}%".format)
        formatted_df["final_syntax_valid"] = formatted_df["final_syntax_valid"].map(lambda x: "Yes" if x else "No")
        formatted_df["final_cis_compliant"] = formatted_df["final_cis_compliant"].map(lambda x: "Yes" if x else "No")
        
        # Rename columns for better display
        formatted_df = formatted_df.rename(columns={
            "run_type": "Run Type",
            "query": "Query",
            "iterations": "Iterations",
            "initial_errors": "Initial Errors",
            "final_errors": "Final Errors",
            "error_reduction": "Error Reduction",
            "error_reduction_pct": "Error Reduction %",
            "initial_compliance_issues": "Initial CIS Issues",
            "final_compliance_issues": "Final CIS Issues",
            "compliance_improvement": "CIS Improvement",
            "compliance_improvement_pct": "CIS Improvement %",
            "final_syntax_valid": "Syntax Valid",
            "final_cis_compliant": "CIS Compliant"
        })
        
        # Generate HTML
        html = f"""
        <html>
        <head>
            <title>Feedback Loop Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                .improvement {{ background-color: #e6ffe6; }}
                .summary {{ margin-top: 30px; }}
            </style>
        </head>
        <body>
            <h1>Feedback Loop Analysis Report</h1>
            <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary of Error Reduction</h2>
            {formatted_df.to_html(index=False)}
            
            <div class="summary">
                <h2>Feedback Loop Performance</h2>
                <p>Average error reduction: {summary_df["error_reduction_pct"].mean():.1f}%</p>
                <p>Average CIS compliance improvement: {summary_df["compliance_improvement_pct"].mean():.1f}%</p>
                <p>Scenarios with syntax validation success: {summary_df["final_syntax_valid"].sum()} of {len(summary_df)}</p>
                <p>Scenarios with CIS compliance success: {summary_df["final_cis_compliant"].sum()} of {len(summary_df)}</p>
            </div>
            
            <h2>Visualization</h2>
            <p>Error reduction charts are available in the same directory as this report.</p>
        </body>
        </html>
        """
        
        # Write HTML to file
        report_file = os.path.join(output_dir, "feedback_analysis_report.html")
        with open(report_file, "w") as f:
            f.write(html)
            
        print(f"Generated summary report: {report_file}")
        
        # Also save as CSV
        csv_file = os.path.join(output_dir, "feedback_analysis_summary.csv")
        summary_df.to_csv(csv_file, index=False)
        print(f"Generated CSV summary: {csv_file}")

def main():
    """Main function to analyze feedback logs"""
    print("Analyzing feedback logs...")
    
    # Get logs grouped by run
    logs_by_run = get_logs_by_run()
    
    if not logs_by_run:
        print("No feedback logs found. Run test_feedback_loop.py first.")
        return
        
    print(f"Found {len(logs_by_run)} unique test runs")
    
    # Analyze error progression
    df = analyze_error_progression(logs_by_run)
    
    if df.empty:
        print("No data to analyze")
        return
        
    # Generate charts
    generate_error_charts(df)
    
    # Generate summary report
    generate_summary_report(df)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 