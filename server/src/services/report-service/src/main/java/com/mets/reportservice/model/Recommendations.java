package com.mets.reportservice.model;

import lombok.Data;

import java.util.List;

@Data
public class Recommendations {

    private List<String> dietPlan;
    private List<String> avoidList;
    private List<String> exercisePlan;
    private List<String> yogaPoses;
}
