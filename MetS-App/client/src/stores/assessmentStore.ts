/**
 * Assessment Store (Zustand)
 * Manages health assessment flow, form data, results & recommendations
 */

import { create } from 'zustand';
import type {
    PatientInfo,
    AdditionalInfo,
    AssessmentResults,
    Recommendations,
    AssessmentInputParameters,
} from '../data/types';
import {
    predictMetabolicSyndrome,
    calculateSeverity,
    getRecommendations,
    generateReport,
    downloadReport,
    saveAssessment,
} from '../data/api';

// ============ Default Values ============

const DEFAULT_PATIENT_INFO: PatientInfo = {
    age: 30,
    gender: 'Men',
    fattyLiver: false,
    hypertension: false,
    diabetes: false,
    systolicBP: 120,
    diastolicBP: 80,
    waistCircumference: 75,
};

const DEFAULT_ADDITIONAL_INFO: AdditionalInfo = {
    hdlCholesterol: 50,
    triglyceride: 150,
    fpg: 90,
};

const DEFAULT_RESULTS: AssessmentResults = {
    probability: 0,
    hasMetabolicSyndrome: false,
};

const DEFAULT_RECOMMENDATIONS: Recommendations = {
    dietPlan: [],
    avoidList: [],
    exercisePlan: [],
    yogaPoses: [],
};

// ============ Store Interface ============

interface AssessmentState {
    // Form state
    activeStep: number;
    loading: boolean;
    error: string | null;
    tabValue: number;
    termsAccepted: boolean;
    termsOpen: boolean;
    termsCheckbox: boolean;

    // Data
    patientInfo: PatientInfo;
    additionalInfo: AdditionalInfo;
    results: AssessmentResults;
    recommendations: Recommendations;

    // Actions — Form
    setPatientInfo: (field: keyof PatientInfo, value: unknown) => void;
    setAdditionalInfo: (field: keyof AdditionalInfo, value: number) => void;
    setTabValue: (value: number) => void;
    setError: (error: string | null) => void;
    setActiveStep: (step: number) => void;
    setTermsCheckbox: (value: boolean) => void;
    acceptTerms: () => void;

    // Actions — API flows
    predict: () => Promise<void>;
    calculateSeverity: () => Promise<void>;
    downloadReport: () => Promise<void>;
    saveCurrentAssessment: () => Promise<void>;

    // Actions — Reset
    startNewAssessment: () => void;
}

// ============ Helper ============

const buildInputParameters = (
    patientInfo: PatientInfo,
    additionalInfo: AdditionalInfo,
    hasMetabolicSyndrome: boolean
): AssessmentInputParameters => ({
    age: patientInfo.age,
    gender: patientInfo.gender,
    fattyLiver: patientInfo.fattyLiver,
    hypertension: patientInfo.hypertension,
    diabetes: patientInfo.diabetes,
    systolicBP: patientInfo.systolicBP,
    diastolicBP: patientInfo.diastolicBP,
    waistCircumference: patientInfo.waistCircumference,
    ...(hasMetabolicSyndrome ? {
        hdlCholesterol: additionalInfo.hdlCholesterol,
        triglyceride: additionalInfo.triglyceride,
        fpg: additionalInfo.fpg,
    } : {}),
});

// ============ Store ============

export const useAssessmentStore = create<AssessmentState>((set, get) => ({
    // Initial state
    activeStep: 0,
    loading: false,
    error: null,
    tabValue: 0,
    termsAccepted: false,
    termsOpen: true,
    termsCheckbox: false,

    patientInfo: { ...DEFAULT_PATIENT_INFO },
    additionalInfo: { ...DEFAULT_ADDITIONAL_INFO },
    results: { ...DEFAULT_RESULTS },
    recommendations: { ...DEFAULT_RECOMMENDATIONS },

    // ---- Form actions ----

    setPatientInfo: (field, value) =>
        set((s) => ({ patientInfo: { ...s.patientInfo, [field]: value } })),

    setAdditionalInfo: (field, value) =>
        set((s) => ({ additionalInfo: { ...s.additionalInfo, [field]: value } })),

    setTabValue: (value) => set({ tabValue: value }),
    setError: (error) => set({ error }),
    setActiveStep: (step) => set({ activeStep: step }),
    setTermsCheckbox: (value) => set({ termsCheckbox: value }),

    acceptTerms: () => {
        if (get().termsCheckbox) {
            set({ termsAccepted: true, termsOpen: false });
        }
    },

    // ---- API flows ----

    predict: async () => {
        const { patientInfo } = get();
        set({ loading: true, error: null });

        try {
            const { probability, hasMetabolicSyndrome } = await predictMetabolicSyndrome(patientInfo);
            set({ results: { probability, hasMetabolicSyndrome } });

            if (hasMetabolicSyndrome) {
                set({ activeStep: 1 }); // → Additional Info
            } else {
                const recs = await getRecommendations(patientInfo.gender, 'Low Severity', patientInfo.age);
                set({ recommendations: recs, activeStep: 1 }); // → Results (2-step flow)
            }
        } catch (err) {
            console.error('Prediction error:', err);
            set({ error: 'Failed to get prediction. Please make sure the server and Python prediction service are running.' });
        } finally {
            set({ loading: false });
        }
    },

    calculateSeverity: async () => {
        const { patientInfo, additionalInfo, results } = get();
        set({ loading: true, error: null });

        try {
            const { severity, riskLevel } = await calculateSeverity(patientInfo, additionalInfo, results.probability);
            set({ results: { ...results, severity, riskLevel } });

            const recs = await getRecommendations(patientInfo.gender, riskLevel, patientInfo.age);
            set({ recommendations: recs, activeStep: 2 }); // → Results (3-step flow)
        } catch (err) {
            console.error('Severity error:', err);
            set({ error: 'Failed to calculate severity. Please try again.' });
        } finally {
            set({ loading: false });
        }
    },

    downloadReport: async () => {
        const { patientInfo, results, recommendations } = get();

        try {
            const userInfo = {
                Age: patientInfo.age,
                Gender: patientInfo.gender,
                'Fatty Liver': patientInfo.fattyLiver ? 'Yes' : 'No',
                Hypertension: patientInfo.hypertension ? 'Yes' : 'No',
                Diabetes: patientInfo.diabetes ? 'Yes' : 'No',
                'Systolic BP': `${patientInfo.systolicBP} mmHg`,
                'Diastolic BP': `${patientInfo.diastolicBP} mmHg`,
                'Waist Circumference': `${patientInfo.waistCircumference} cm`,
            };

            const { report } = await generateReport({ userInfo, results, recommendations });
            const date = new Date().toISOString().split('T')[0];
            downloadReport(report, `metabolic_syndrome_health_plan_${date}.md`);
        } catch (err) {
            console.error('Report error:', err);
            set({ error: 'Failed to generate report. Please try again.' });
        }
    },

    saveCurrentAssessment: async () => {
        const { patientInfo, additionalInfo, results, recommendations } = get();
        try {
            const inputParameters = buildInputParameters(patientInfo, additionalInfo, results.hasMetabolicSyndrome);
            await saveAssessment({
                probability: results.probability,
                severity: results.severity || 0,
                riskLevel: results.riskLevel || 'Low Severity',
                recommendations,
                inputParameters,
            });
            set({ error: null });
        } catch (err) {
            console.error('Failed to save assessment:', err);
            // The API interceptor will handle 401 redirect to login
            if ((err as any).response?.status !== 401) {
                set({ error: 'Failed to save assessment. Please try again.' });
            }
        }
    },

    // ---- Reset ----

    startNewAssessment: () =>
        set({
            activeStep: 0,
            loading: false,
            error: null,
            tabValue: 0,
            patientInfo: { ...DEFAULT_PATIENT_INFO },
            additionalInfo: { ...DEFAULT_ADDITIONAL_INFO },
            results: { ...DEFAULT_RESULTS },
            recommendations: { ...DEFAULT_RECOMMENDATIONS },
        }),
}));
