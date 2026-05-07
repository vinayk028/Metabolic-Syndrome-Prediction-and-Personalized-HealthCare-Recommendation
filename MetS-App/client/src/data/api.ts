/**
 * API Service - All API calls in one place
 */

import axios from 'axios';
import type {
    PatientInfo,
    AdditionalInfo,
    PredictionResult,
    SeverityResult,
    Recommendations,
    ReportData,
    AuthResponse,
    SignupData,
    LoginData,
    UpdateProfileData,
    UpdatePasswordData,
    User,
    NewsResponse,
    AssessmentHistoryItem,
    AssessmentInputParameters,
    ChatResponse,
} from './types';

// API Base URL
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

// Create axios instance
const api = axios.create({
    baseURL: API_URL,
    headers: { 'Content-Type': 'application/json' },
});

// Add auth token to requests
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// Handle 401 errors (redirect to login)
api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            localStorage.removeItem('token');
            localStorage.removeItem('user');
            if (!['/login', '/signup'].includes(window.location.pathname)) {
                window.location.href = '/login';
            }
        }
        return Promise.reject(error);
    }
);

// ============ Auth APIs ============

export const signup = async (data: SignupData): Promise<AuthResponse> => {
    const response = await api.post('/auth/signup', data);
    return response.data;
};

export const login = async (data: LoginData): Promise<AuthResponse> => {
    const response = await api.post('/auth/login', data);
    return response.data;
};

export const getCurrentUser = async (): Promise<{ success: boolean; user: User }> => {
    const response = await api.get('/auth/me');
    return response.data;
};

export const updateProfile = async (data: UpdateProfileData): Promise<AuthResponse> => {
    const response = await api.put('/auth/profile', data);
    return response.data;
};

export const updatePassword = async (data: UpdatePasswordData): Promise<AuthResponse> => {
    const response = await api.put('/auth/password', data);
    return response.data;
};

export const deleteAccount = async (): Promise<AuthResponse> => {
    const response = await api.delete('/auth/account');
    return response.data;
};

// ============ Assessment APIs ============

export const healthCheck = async () => {
    const response = await api.get('/health');
    return response.data;
};

export const predictMetabolicSyndrome = async (patientInfo: PatientInfo): Promise<PredictionResult> => {
    const response = await api.post('/mets/predict', patientInfo);
    return response.data;
};

export const calculateSeverity = async (
    patientInfo: PatientInfo,
    additionalInfo: AdditionalInfo,
    probability: number
): Promise<SeverityResult> => {
    const response = await api.post('/mets/severity', {
        gender: patientInfo.gender,
        age: patientInfo.age,
        systolicBP: patientInfo.systolicBP,
        waistCircumference: patientInfo.waistCircumference,
        fpg: additionalInfo.fpg,
        triglyceride: additionalInfo.triglyceride,
        hdlCholesterol: additionalInfo.hdlCholesterol,
        probability,
    });
    return response.data;
};

export const getRecommendations = async (
    gender: string,
    riskLevel: string,
    age: number
): Promise<Recommendations> => {
    const response = await api.post('/mets/recommendations', { gender, riskLevel, age });
    return response.data;
};

export const generateReport = async (reportData: ReportData): Promise<{ report: string }> => {
    const response = await api.post('/mets/report', reportData);
    return response.data;
};

export const saveAssessment = async (data: {
    probability: number;
    severity: number;
    riskLevel: string;
    recommendations: Recommendations;
    inputParameters?: AssessmentInputParameters;
}): Promise<{ success: boolean; message: string }> => {
    const response = await api.post('/auth/assessment', data);
    return response.data;
};

export const getAssessmentHistory = async (): Promise<{ 
    success: boolean; 
    assessmentHistory: AssessmentHistoryItem[] 
}> => {
    const response = await api.get('/auth/assessments');
    return response.data;
};

// ============ News APIs ============

export const getMetabolicSyndromeNews = async (limit: number = 15, skip: number = 0): Promise<NewsResponse> => {
    const response = await api.get(`/news?limit=${limit}&skip=${skip}`);
    return response.data;
};

// ============ Chat APIs ============

export const sendChatMessage = async (
    message: string,
    history: { role: string; content: string }[]
): Promise<ChatResponse> => {
    const response = await api.post('/chat', { message, history });
    return response.data;
};

// ============ Helpers ============

export const downloadReport = (report: string, filename: string): void => {
    // Decode base64 PDF to binary
    const binaryString = window.atob(report);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    
    // Create blob from binary data
    const blob = new Blob([bytes], { type: 'application/pdf' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename.replace('.md', '.pdf');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
};

export default api;
