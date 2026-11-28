import { describe, expect, it } from 'vitest';
import { getFactorStatus, toRiskKey } from '@/theme/factors';

describe('getFactorStatus', () => {
  it('uses gender-specific waist cut-offs', () => {
    expect(getFactorStatus('waist', 101, 'Men')).toBe('borderline');
    expect(getFactorStatus('waist', 102, 'Men')).toBe('elevated');
    expect(getFactorStatus('waist', 93, 'Men')).toBe('normal');
    expect(getFactorStatus('waist', 88, 'Women')).toBe('elevated');
    expect(getFactorStatus('waist', 80, 'Women')).toBe('borderline');
    expect(getFactorStatus('waist', 79, 'Women')).toBe('normal');
  });

  it('flags blood pressure on either systolic or diastolic', () => {
    expect(getFactorStatus('bloodPressure', 118, 'Men', 76)).toBe('normal');
    expect(getFactorStatus('bloodPressure', 118, 'Men', 82)).toBe('borderline');
    expect(getFactorStatus('bloodPressure', 125, 'Men', 70)).toBe('borderline');
    expect(getFactorStatus('bloodPressure', 118, 'Men', 85)).toBe('elevated');
    expect(getFactorStatus('bloodPressure', 130, 'Men', 70)).toBe('elevated');
  });

  it('classifies glucose and triglycerides at their thresholds', () => {
    expect(getFactorStatus('glucose', 89)).toBe('normal');
    expect(getFactorStatus('glucose', 90)).toBe('borderline');
    expect(getFactorStatus('glucose', 100)).toBe('elevated');
    expect(getFactorStatus('triglycerides', 119)).toBe('normal');
    expect(getFactorStatus('triglycerides', 120)).toBe('borderline');
    expect(getFactorStatus('triglycerides', 150)).toBe('elevated');
  });

  it('treats low HDL as the risk, with a higher bar for women', () => {
    expect(getFactorStatus('hdl', 39, 'Men')).toBe('elevated');
    expect(getFactorStatus('hdl', 45, 'Men')).toBe('borderline');
    expect(getFactorStatus('hdl', 50, 'Men')).toBe('normal');
    expect(getFactorStatus('hdl', 45, 'Women')).toBe('elevated');
    expect(getFactorStatus('hdl', 55, 'Women')).toBe('borderline');
    expect(getFactorStatus('hdl', 60, 'Women')).toBe('normal');
  });
});

describe('toRiskKey', () => {
  it('maps backend severity strings to token keys', () => {
    expect(toRiskKey('Low Severity')).toBe('low');
    expect(toRiskKey('Medium Severity')).toBe('medium');
    expect(toRiskKey('Moderate')).toBe('medium');
    expect(toRiskKey('HIGH Severity')).toBe('high');
  });

  it('returns null for missing or unknown values', () => {
    expect(toRiskKey(undefined)).toBeNull();
    expect(toRiskKey(null)).toBeNull();
    expect(toRiskKey('')).toBeNull();
    expect(toRiskKey('unknown')).toBeNull();
  });
});
