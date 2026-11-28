import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/services/api', () => ({
  predictMetabolicSyndrome: vi.fn(),
  calculateSeverity: vi.fn(),
  getRecommendations: vi.fn(),
  generateReport: vi.fn(),
  downloadReport: vi.fn(),
  saveAssessment: vi.fn().mockResolvedValue({}),
}));

import { saveAssessment } from '@/services/api';
import { useAssessmentStore } from '@/stores/assessmentStore';

describe('saveCurrentAssessment', () => {
  beforeEach(() => {
    vi.mocked(saveAssessment).mockClear();
    useAssessmentStore.getState().startNewAssessment();
  });

  it('does nothing for guests (no token), so they keep their result', async () => {
    await useAssessmentStore.getState().saveCurrentAssessment();
    expect(saveAssessment).not.toHaveBeenCalled();
  });

  it('saves for signed-in users, without blood tests when not needed', async () => {
    localStorage.setItem('token', 'test-token');
    await useAssessmentStore.getState().saveCurrentAssessment();
    expect(saveAssessment).toHaveBeenCalledTimes(1);
    const payload = vi.mocked(saveAssessment).mock.calls[0][0];
    expect(payload.inputParameters).toMatchObject({ gender: 'Men', systolicBP: 120 });
    expect(payload.inputParameters).not.toHaveProperty('fpg');
    expect(payload.riskLevel).toBe('Low Severity');
  });

  it('includes blood tests when the result needs them', async () => {
    localStorage.setItem('token', 'test-token');
    useAssessmentStore.setState({ results: { probability: 0.8, hasMetabolicSyndrome: true } });
    await useAssessmentStore.getState().saveCurrentAssessment();
    const payload = vi.mocked(saveAssessment).mock.calls[0][0];
    expect(payload.inputParameters).toMatchObject({ fpg: 90, triglyceride: 150, hdlCholesterol: 50 });
  });
});
