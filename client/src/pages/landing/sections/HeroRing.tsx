/**
 * HeroRing — the five metabolic-syndrome markers as a dashed ring. It walks
 * through the markers on its own (paused on hover/focus and for reduced
 * motion); hover or tap a marker to pin it.
 */
import { useEffect, useState } from 'react';
import { Typography } from '@mui/material';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { MetabolicRing } from '../../../components/ui';
import { FACTORS, FACTOR_ORDER } from '../../../theme/factors';
import { factorVar } from '../../../theme/cssVars';
import type { FactorKey } from '../../../theme/tokens';

const AUTO_MS = 3200;

const HeroRing = () => {
  const reduce = useReducedMotion();
  const [active, setActive] = useState<FactorKey>(FACTOR_ORDER[0]);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    if (paused || reduce) return;
    const id = window.setInterval(() => {
      setActive((cur) => FACTOR_ORDER[(FACTOR_ORDER.indexOf(cur) + 1) % FACTOR_ORDER.length]);
    }, AUTO_MS);
    return () => window.clearInterval(id);
  }, [paused, reduce]);

  const F = FACTORS[active];
  const Icon = F.icon;

  return (
    <div
      className="lp-ring"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
    >
      <div className="lp-ring__disc">
        <MetabolicRing colorBy="factor" size={400} thickness={14} activeFactor={active} onFactorSelect={setActive}>
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={active}
              className="lp-ring__center"
              initial={reduce ? false : { opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reduce ? { opacity: 0 } : { opacity: 0, y: -6 }}
              transition={{ duration: 0.22 }}
            >
              <span
                className="lp-ring__icon"
                style={{ color: factorVar(active), background: `color-mix(in srgb, ${factorVar(active)} 14%, transparent)` }}
              >
                <Icon aria-hidden />
              </span>
              <Typography component="span" className="lp-ring__name">
                {F.label}
              </Typography>
              <span className="lp-ring__threshold">{F.threshold}</span>
            </motion.div>
          </AnimatePresence>
        </MetabolicRing>
      </div>

      <ul className="lp-ring__chips" aria-label="The five markers">
        {FACTOR_ORDER.map((k) => {
          const CI = FACTORS[k].icon;
          const on = k === active;
          return (
            <li key={k}>
              <button
                type="button"
                className={`lp-ring__chip${on ? ' is-active' : ''}`}
                aria-pressed={on}
                onClick={() => setActive(k)}
                onMouseEnter={() => setActive(k)}
                style={{ ['--chip' as string]: factorVar(k) }}
              >
                <CI aria-hidden />
                {FACTORS[k].shortLabel}
              </button>
            </li>
          );
        })}
      </ul>

      <p className="lp-ring__note">
        When <strong>three or more</strong> are out of range, the risk of type 2 diabetes, heart disease and stroke rises
        sharply.
      </p>
    </div>
  );
};

export default HeroRing;
