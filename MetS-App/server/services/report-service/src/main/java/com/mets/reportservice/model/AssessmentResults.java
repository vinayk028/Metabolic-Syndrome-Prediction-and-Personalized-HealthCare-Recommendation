package com.mets.reportservice.model;

import lombok.Data;

@Data
public class AssessmentResults {

    private double probability;
    private Double severity;
    private String riskLevel;
}
