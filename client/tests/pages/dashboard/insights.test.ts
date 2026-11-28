import { afterEach, describe, expect, it, vi } from 'vitest';
import type { AssessmentHistoryItem } from '@/types/types';
import { RECHECK_DAYS, buildMarkerSeries, nextCheck } from '@/pages/dashboard/insights';

const item = (date: string, p: Partial<NonNullable<AssessmentHistoryItem['inputParameters']>> | undefined) =>
  ({
    _id: date,
    date,
    probability: 0.4,
    severity: 0,
    riskLevel: 'Low Severity',
    recommendations: { dietPlan: [], avoidList: [], exercisePlan: [], yogaPoses: [] },
    inputParameters: p && {
      age: 40,
      gender: 'Men',
      fattyLiver: false,
      hypertension: false,
      diabetes: false,
      systolicBP: 120,
      diastolicBP: 80,
      waistCircumference: 100,
      ...p,
    },
  }) as AssessmentHistoryItem;

describe('buildMarkerSeries', () => {
  it('collects one point per assessment and skips missing values', () => {
    const s = buildMarkerSeries([
      item('2026-01-01', { waistCircumference: 104, fpg: 110 }),
      item('2026-04-01', { waistCircumference: 98 }),
      item('2026-07-01', undefined),
    ]);
    expect(s.waist!.points.map((p) => p.value)).toEqual([104, 98]);
    expect(s.glucose!.points).toHaveLength(1);
    expect(s.triglycerides).toBeUndefined();
  });

  it('reports the change and whether it is an improvement', () => {
    const s = buildMarkerSeries([
      item('2026-01-01', { waistCircumference: 104, systolicBP: 135, hdlCholesterol: 38 }),
      item('2026-04-01', { waistCircumference: 98, systolicBP: 135, hdlCholesterol: 44 }),
    ]);
    expect(s.waist!.delta).toBe(-6);
    expect(s.waist!.improving).toBe(true); // lower waist is better
    expect(s.hdl!.delta).toBe(6);
    expect(s.hdl!.improving).toBe(true); // higher HDL is better
    expect(s.bloodPressure!.improving).toBeNull(); // no change
  });

  it('flags a worsening marker', () => {
    const s = buildMarkerSeries([item('2026-01-01', { fpg: 95 }), item('2026-04-01', { fpg: 104 })]);
    expect(s.glucose!.improving).toBe(false);
  });

  it('shows blood pressure as systolic/diastolic', () => {
    const s = buildMarkerSeries([item('2026-01-01', { systolicBP: 128, diastolicBP: 84 })]);
    expect(s.bloodPressure!.points[0].display).toBe('128/84');
    expect(s.bloodPressure!.improving).toBeNull(); // a single point has no trend
  });
});

describe('nextCheck', () => {
  afterEach(() => vi.useRealTimers());

  it(`is due ${RECHECK_DAYS} days after the last check`, () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-03-01T00:00:00Z'));
    const { due, daysLeft } = nextCheck('2026-01-01T00:00:00Z');
    expect(due.toISOString()).toBe('2026-04-01T00:00:00.000Z');
    expect(daysLeft).toBe(31);
  });

  it('goes negative once the check is overdue', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-05-01T00:00:00Z'));
    expect(nextCheck('2026-01-01T00:00:00Z').daysLeft).toBeLessThan(0);
  });
});
