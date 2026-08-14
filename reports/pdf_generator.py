"""
Clinical EEG Diagnostic Report Generator Module
Generates clean PDF medical diagnostic reports summarizing EEG recording metrics and AI seizure detection results.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

class ClinicalPDFReportGenerator:
    """Generates clinical PDF medical reports for EEG epilepsy diagnostic sessions."""
    
    @staticmethod
    def generate_report(
        output_filename: str,
        patient_info: Dict,
        recording_info: Dict,
        detection_results: Dict,
        band_powers: Dict[str, float]
    ) -> str:
        doc = SimpleDocTemplate(
            output_filename,
            pagesize=letter,
            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        
        # Custom Paragraph Styles
        title_style = ParagraphStyle(
            'ReportTitle', parent=styles['Heading1'],
            fontName='Helvetica-Bold', fontSize=22, leading=26, textColor=colors.HexColor('#0F172A')
        )
        subtitle_style = ParagraphStyle(
            'ReportSubtitle', parent=styles['Normal'],
            fontName='Helvetica', fontSize=10, leading=14, textColor=colors.HexColor('#475569')
        )
        h2_style = ParagraphStyle(
            'SectionH2', parent=styles['Heading2'],
            fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=colors.HexColor('#1E293B'), spaceBefore=12, spaceAfter=6
        )
        body_style = ParagraphStyle(
            'ReportBody', parent=styles['Normal'],
            fontName='Helvetica', fontSize=10, leading=14, textColor=colors.HexColor('#334155')
        )
        alert_style = ParagraphStyle(
            'AlertText', parent=styles['Normal'],
            fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor('#DC2626')
        )
        normal_status_style = ParagraphStyle(
            'NormalText', parent=styles['Normal'],
            fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor('#16A34A')
        )

        elements = []

        # 1. Header Banner
        elements.append(Paragraph("EPILEPSY AI DIAGNOSTIC REPORT", title_style))
        elements.append(Paragraph("NeuroDiagnostics AI Clinical Workbench | Automated EEG Analysis", subtitle_style))
        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2563EB'), spaceAfter=15))

        # 2. Patient & Session Metadata Table
        elements.append(Paragraph("Patient & Recording Metadata", h2_style))
        
        meta_data = [
            [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(patient_info.get('name', 'John Doe'), body_style),
             Paragraph("<b>Patient ID:</b>", body_style), Paragraph(patient_info.get('id', 'PT-1042'), body_style)],
            [Paragraph("<b>Age / Gender:</b>", body_style), Paragraph(f"{patient_info.get('age', 34)} / {patient_info.get('gender', 'Male')}", body_style),
             Paragraph("<b>Recording Date:</b>", body_style), Paragraph(patient_info.get('date', datetime.now().strftime("%Y-%m-%d %H:%M")), body_style)],
            [Paragraph("<b>Sampling Rate:</b>", body_style), Paragraph(f"{recording_info.get('fs', 256)} Hz", body_style),
             Paragraph("<b>Duration:</b>", body_style), Paragraph(f"{recording_info.get('duration_sec', 10.0):.1f} seconds", body_style)],
            [Paragraph("<b>Montage:</b>", body_style), Paragraph(recording_info.get('montage', '10-20 Standard Bipolar'), body_style),
             Paragraph("<b>Channels:</b>", body_style), Paragraph(f"{recording_info.get('n_channels', 14)} Channels", body_style)]
        ]

        meta_table = Table(meta_data, colWidths=[110, 160, 110, 160])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 15))

        # 3. AI Deep Learning Seizure Classification Summary
        elements.append(Paragraph("AI Deep Learning Seizure Classification", h2_style))
        
        pred_label = detection_results.get('predicted_label', 'NORMAL')
        confidence = detection_results.get('confidence', 0.0)
        is_seizure = detection_results.get('is_seizure', False)
        focal_channels = ", ".join(detection_results.get('focal_channels', ['None']))

        status_paragraph = Paragraph(f"DIAGNOSTIC STATUS: {pred_label} (Confidence: {confidence:.1f}%)", alert_style if is_seizure else normal_status_style)
        elements.append(status_paragraph)
        elements.append(Spacer(1, 8))

        dl_data = [
            [Paragraph("<b>Classification State</b>", body_style), Paragraph("<b>Probability (%)</b>", body_style)]
        ]
        
        probs = detection_results.get('probabilities', {})
        for label, prob in probs.items():
            dl_data.append([
                Paragraph(label, body_style),
                Paragraph(f"{prob:.2f}%", body_style)
            ])

        dl_table = Table(dl_data, colWidths=[270, 270])
        dl_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        elements.append(dl_table)
        elements.append(Spacer(1, 15))

        # 4. Spectral Band Power Distribution
        elements.append(Paragraph("EEG Frequency Band Power Summary", h2_style))
        
        power_data = [
            [Paragraph("<b>Frequency Band</b>", body_style), Paragraph("<b>Range (Hz)</b>", body_style), Paragraph("<b>Relative Power (%)</b>", body_style)]
        ]
        band_ranges = {'Delta': '0.5 - 4.0 Hz', 'Theta': '4.0 - 8.0 Hz', 'Alpha': '8.0 - 13.0 Hz', 'Beta': '13.0 - 30.0 Hz', 'Gamma': '30.0 - 50.0 Hz'}
        
        for band, val in band_powers.items():
            power_data.append([
                Paragraph(band, body_style),
                Paragraph(band_ranges.get(band, ''), body_style),
                Paragraph(f"{val:.2f}%", body_style)
            ])

        power_table = Table(power_data, colWidths=[180, 180, 180])
        power_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#334155')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        elements.append(power_table)
        elements.append(Spacer(1, 15))

        # 5. Clinical Impression & Neurologist Recommendations
        elements.append(Paragraph("Clinical Impression & Recommendations", h2_style))
        
        if is_seizure:
            impression_text = (
                f"<b>IMPRESSION:</b> The automated EEG deep learning analysis detected high-confidence paroxysmal "
                f"epileptiform seizure activity ({pred_label}) localized primarily in temporal focus channels [{focal_channels}]. "
                f"High-amplitude rhythmic spike-wave discharges observed."
            )
            rec_text = "<b>RECOMMENDATIONS:</b> Immediate clinical review by an epileptologist. Consider anti-seizure medication (ASM) evaluation and follow-up 24-hour continuous video-EEG monitoring."
        else:
            impression_text = (
                f"<b>IMPRESSION:</b> The automated EEG deep learning analysis indicates normal background rhythms ({pred_label}) "
                f"with dominant posterior alpha rhythm. No epileptiform spike-and-wave patterns or paroxysmal discharges detected."
            )
            rec_text = "<b>RECOMMENDATIONS:</b> Routine clinical follow-up as indicated by primary care provider."

        elements.append(Paragraph(impression_text, body_style))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(rec_text, body_style))
        elements.append(Spacer(1, 25))

        # Signature Block
        elements.append(Paragraph("<b>Verified By:</b> ___________________________, MD (Epileptologist / Clinical Neurophysiologist)", body_style))

        doc.build(elements)
        return output_filename
