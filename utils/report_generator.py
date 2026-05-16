from fpdf import FPDF
import tempfile
import os

def generate_prediction_pdf(passenger_details, prediction, confidence, figures=None):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.set_fill_color(0, 102, 204) # Blue header
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, " TITANIC SURVIVAL PREDICTION ANALYSIS ", 0, 1, 'C', 1)
    pdf.ln(5)
    
    # Passenger Details
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Passenger Profile:", 0, 1)
    
    pdf.set_font("Arial", '', 11)
    details = f"""
Age: {passenger_details['Age']}
Gender: {passenger_details['Sex']}
Passenger Class: {passenger_details['Pclass']}
Fare: {passenger_details['Fare']}
Siblings/Spouses Aboard: {passenger_details['SibSp']}
Parents/Children Aboard: {passenger_details['Parch']}
Port of Embarkation: {passenger_details['Embarked']}
    """
    for line in details.strip().split('\n'):
        pdf.cell(0, 6, line.strip(), 0, 1)
    pdf.ln(5)
    
    # Prediction Result
    pdf.set_font("Arial", 'B', 14)
    result_text = "SURVIVED" if prediction == 1 else "NOT SURVIVED"
    if prediction == 1:
        pdf.set_text_color(0, 150, 0)
    else:
        pdf.set_text_color(200, 0, 0)
        
    pdf.cell(0, 10, f"PREDICTION: {result_text}", 0, 1)
    pdf.cell(0, 10, f"CONFIDENCE: {confidence:.2f}%", 0, 1)
    pdf.ln(10)
    
    # Charts
    if figures:
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, "Global Dataset Analytics:", 0, 1)
        
        temp_dir = tempfile.gettempdir()
        
        for i, fig in enumerate(figures):
            temp_path = os.path.join(temp_dir, f"chart_{i}.png")
            try:
                # Update layout for light background in PDF
                fig.update_layout(template='plotly_white')
                fig.write_image(temp_path, width=400, height=300)
                
                if i % 2 == 0:
                    if pdf.get_y() > 220:
                        pdf.add_page()
                    pdf.image(temp_path, x=10, y=pdf.get_y(), w=90)
                else:
                    pdf.image(temp_path, x=105, y=pdf.get_y(), w=90)
                    pdf.ln(75) 
            except Exception as e:
                print(f"Error generating chart {i}: {e}")
                
    try:
        # FPDF 1.7.2 returns string from output(dest='S'), we encode to bytes
        pdf_str = pdf.output(dest='S')
        if isinstance(pdf_str, str):
            return pdf_str.encode('latin-1')
        return pdf_str
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None
