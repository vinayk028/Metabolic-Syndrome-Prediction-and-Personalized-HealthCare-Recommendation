/**
 * Type Definitions - All types in one place
 */

// ============ Patient & Assessment Types ============

export interface PatientInfo {
    age: number;
    gender: 'Men' | 'Women';
    fattyLiver: boolean;
    hypertension: boolean;
    diabetes: boolean;
    systolicBP: number;
    diastolicBP: number;
    waistCircumference: number;
}

export interface AdditionalInfo {
    hdlCholesterol: number;
    triglyceride: number;
    fpg: number;
}

export interface PredictionResult {
    probability: number;
    hasMetabolicSyndrome: boolean;
}

export interface SeverityResult {
    severity: number;
    riskLevel: string;
}

export interface Recommendations {
    dietPlan: string[];
    avoidList: string[];
    exercisePlan: string[];
    yogaPoses: string[];
}

export interface AssessmentResults {
    probability: number;
    hasMetabolicSyndrome: boolean;
    severity?: number;
    riskLevel?: string;
}

export interface ReportData {
    userInfo: Record<string, string | number>;
    results: AssessmentResults;
    recommendations: Recommendations;
}

export interface AssessmentInputParameters {
    age: number;
    gender: string;
    fattyLiver: boolean;
    hypertension: boolean;
    diabetes: boolean;
    systolicBP: number;
    diastolicBP: number;
    waistCircumference: number;
    hdlCholesterol?: number;
    triglyceride?: number;
    fpg?: number;
}

// ============ User & Auth Types ============

export interface User {
    id: string;
    firstName: string;
    lastName: string;
    email: string;
    fullName: string;
    phone?: string;
    dateOfBirth?: string;
    gender?: string;
    address?: string;
    role: 'user' | 'admin';
    lastLogin?: string;
    assessmentHistory?: AssessmentHistoryItem[];
    createdAt: string;
}

export interface AssessmentHistoryItem {
    _id: string;
    date: string;
    probability: number;
    severity: number;
    riskLevel: string;
    inputParameters?: AssessmentInputParameters;
    recommendations: Recommendations;
}

export interface AuthResponse {
    success: boolean;
    message: string;
    token?: string;
    user?: User;
    errors?: { msg: string; path: string }[];
}

export interface SignupData {
    firstName: string;
    lastName: string;
    email: string;
    password: string;
    confirmPassword: string;
}

export interface LoginData {
    email: string;
    password: string;
}

export interface UpdateProfileData {
    firstName?: string;
    lastName?: string;
    phone?: string;
    dateOfBirth?: string;
    gender?: string;
    address?: string;
}

export interface UpdatePasswordData {
    currentPassword: string;
    newPassword: string;
    confirmPassword: string;
}

// ============ News Types ============

export interface NewsArticle {
    title: string;
    description: string;
    content: string;
    url: string;
    image: string | null;
    source: string;
    publishedAt: string;
    author: string | null;
}

export interface NewsResponse {
    success: boolean;
    count: number;
    articles: NewsArticle[];
    message?: string;
}

// ============ Chat Types ============

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
}

export interface ChatResponse {
    success: boolean;
    response: string;
    message?: string;
}
