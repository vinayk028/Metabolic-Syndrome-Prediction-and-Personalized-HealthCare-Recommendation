package com.mets.reportservice.model;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.util.Map;

@Data
public class ReportRequest {

    @NotNull(message = "Patient info is required")
    private Map<String, String> patientInfo;

    @NotNull(message = "Assessment results are required")
    private AssessmentResults results;

    @NotNull(message = "Recommendations are required")
    private Recommendations recommendations;
}
