import pandas as pd
import json

class DashboardConnector:
    def export_for_powerbi(self, df: pd.DataFrame, filename: str = "dashboard_data.csv"):
        """Export processed data for Power BI consumption"""
        # Prepare data
        dashboard_data = df.copy()
        
        # Add calculated columns
        dashboard_data['Month'] = pd.to_datetime(dashboard_data['Date']).dt.month
        dashboard_data['Year'] = pd.to_datetime(dashboard_data['Date']).dt.year
        
        # Export to CSV
        dashboard_data.to_csv(filename, index=False)
        
        # Create Power BI template
        template = {
            "version": "1.0",
            "defaultVisualType": "card",
            "dataRoles": [
                {"name": "Category", "kind": "Grouping"},
                {"name": "Values", "kind": "Measure"}
            ]
        }
        
        with open("powerbi_template.json", "w") as f:
            json.dump(template, f, indent=2)