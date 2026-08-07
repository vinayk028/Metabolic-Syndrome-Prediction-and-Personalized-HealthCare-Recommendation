package com.mets.reportservice.service;

import com.itextpdf.kernel.colors.ColorConstants;
import com.itextpdf.kernel.pdf.PdfDocument;
import com.itextpdf.kernel.pdf.PdfWriter;
import com.itextpdf.layout.Document;
import com.itextpdf.layout.element.List;
import com.itextpdf.layout.element.ListItem;
import com.itextpdf.layout.element.Paragraph;
import com.itextpdf.layout.properties.TextAlignment;
import com.mets.reportservice.model.AssessmentResults;
import com.mets.reportservice.model.Recommendations;
import com.mets.reportservice.model.ReportRequest;
import org.springframework.stereotype.Service;

import java.io.ByteArrayOutputStream;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.Map;

/**
 * Generates health reports in PDF format using iText7.
 * Single Responsibility: Only handles PDF content generation.
 */
@Service
public class PdfReportService {

    private static final float TITLE_SIZE = 20f;
    private static final float HEADING_SIZE = 14f;
    private static final float BODY_SIZE = 11f;

    public byte[] generate(ReportRequest request) {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

        try (PdfDocument pdf = new PdfDocument(new PdfWriter(outputStream));
             Document doc = new Document(pdf)) {

            String date = LocalDate.now().format(DateTimeFormatter.ofPattern("MMMM dd, yyyy"));

            // Title
            doc.add(new Paragraph("METABOLIC SYNDROME HEALTH PLAN")
                    .setFontSize(TITLE_SIZE).setBold().setTextAlignment(TextAlignment.CENTER));
            doc.add(new Paragraph("Generated on: " + date)
                    .setFontSize(BODY_SIZE).setTextAlignment(TextAlignment.CENTER));

            addPatientInfo(doc, request.getPatientInfo());
            addResults(doc, request.getResults());
            addRecommendations(doc, request.getRecommendations());
            addDisclaimer(doc);
        }

        return outputStream.toByteArray();
    }

    private void addPatientInfo(Document doc, Map<String, String> info) {
        doc.add(createHeading("PATIENT INFORMATION"));
        List list = new List();
        info.forEach((key, value) -> list.add(new ListItem(key + ": " + value)));
        doc.add(list);
    }

    private void addResults(Document doc, AssessmentResults results) {
        doc.add(createHeading("ASSESSMENT RESULTS"));
        doc.add(new Paragraph(String.format("Probability of Metabolic Syndrome: %.1f%%", results.getProbability() * 100)));

        if (results.getSeverity() != null) {
            doc.add(new Paragraph(String.format("Severity Score: %.2f", results.getSeverity())));
            doc.add(new Paragraph("Risk Level: " + results.getRiskLevel()).setBold());
        }
    }

    private void addRecommendations(Document doc, Recommendations rec) {
        doc.add(createHeading("HEALTH RECOMMENDATIONS"));
        addSection(doc, "Diet Plan", rec.getDietPlan());
        addSection(doc, "Foods to Avoid", rec.getAvoidList());
        addSection(doc, "Exercise Plan", rec.getExercisePlan());
        addSection(doc, "Yoga Poses", rec.getYogaPoses());
    }

    private void addSection(Document doc, String title, java.util.List<String> items) {
        if (items == null || items.isEmpty()) return;
        doc.add(new Paragraph(title).setFontSize(12f).setBold());
        List list = new List();
        items.forEach(item -> list.add(new ListItem(item)));
        doc.add(list);
    }

    private void addDisclaimer(Document doc) {
        doc.add(createHeading("DISCLAIMER"));
        doc.add(new Paragraph(
                "This health plan is for informational purposes only. " +
                "It is not a substitute for professional medical advice. " +
                "Always consult your physician.")
                .setFontSize(9f).setFontColor(ColorConstants.GRAY));
    }

    private Paragraph createHeading(String text) {
        return new Paragraph(text).setFontSize(HEADING_SIZE).setBold().setMarginTop(15f);
    }
}
