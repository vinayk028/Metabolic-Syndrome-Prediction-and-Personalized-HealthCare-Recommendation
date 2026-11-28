/**
 * Accessibility check with axe-core for component tests.
 *
 * Runs on document.body so MUI dialogs and menus (rendered in portals) are
 * included. Rules that need a real browser or a whole page are switched off:
 * colour contrast (jsdom has no layout or colours) and page-level landmark /
 * heading rules (a single component is not a page).
 */
import axe from 'axe-core';

const OFF = ['color-contrast', 'region', 'landmark-one-main', 'page-has-heading-one'];

/** Readable list of violations — empty when the markup passes. */
export const axeViolations = async (root: Element = document.body): Promise<string[]> => {
  const result = await axe.run(root, {
    runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa', 'best-practice'] },
    rules: Object.fromEntries(OFF.map((id) => [id, { enabled: false }])),
    resultTypes: ['violations'],
  });
  return result.violations.map(
    (v) => `${v.id} (${v.impact}): ${v.help}\n    ${v.nodes.map((n) => n.target.join(' ')).join('\n    ')}`,
  );
};
