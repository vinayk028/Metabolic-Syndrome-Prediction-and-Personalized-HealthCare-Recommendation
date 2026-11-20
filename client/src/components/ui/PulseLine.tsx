/**
 * PulseLine — the heartbeat (ECG) motif.
 * Used in the hero, as a section divider and (with `loop`) as a loader.
 */
import { Box, type SxProps, type Theme } from '@mui/material';
import { motion, useReducedMotion } from 'framer-motion';

interface PulseLineProps {
  /** Stroke colour. Defaults to Pulse Coral. */
  color?: string;
  /** Rendered height in px; width fills the container. */
  height?: number;
  strokeWidth?: number;
  /** Continuously redraw (loader). Otherwise draws once when mounted. */
  loop?: boolean;
  /** Seconds for one draw. */
  duration?: number;
  /** Fade the line out at both ends. */
  fadeEdges?: boolean;
  sx?: SxProps<Theme>;
  'aria-label'?: string;
}

// One heartbeat repeated across a 400×60 canvas.
const BEAT_PATH =
  'M0 30 H70 L80 30 L88 22 L96 30 H112 L120 40 L132 4 L144 56 L154 30 H176 L188 22 L200 30 ' +
  'H270 L280 30 L288 22 L296 30 H312 L320 40 L332 4 L344 56 L354 30 H376 L388 22 L400 30';

const PulseLine = ({
  color = 'var(--vb-coral-400)',
  height = 48,
  strokeWidth = 2.5,
  loop = false,
  duration = 2.2,
  fadeEdges = true,
  sx,
  ...rest
}: PulseLineProps) => {
  const reduce = useReducedMotion();
  const label = rest['aria-label'];

  return (
    <Box
      role={label ? 'img' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      sx={[
        {
          width: '100%',
          height,
          color,
          ...(fadeEdges && {
            maskImage: 'linear-gradient(90deg, transparent, #000 12%, #000 88%, transparent)',
            WebkitMaskImage: 'linear-gradient(90deg, transparent, #000 12%, #000 88%, transparent)',
          }),
        },
        ...(Array.isArray(sx) ? sx : [sx]),
      ]}
    >
      <svg viewBox="0 0 400 60" preserveAspectRatio="none" width="100%" height="100%">
        <motion.path
          d={BEAT_PATH}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
          initial={reduce ? false : { pathLength: 0, opacity: 0.4 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={
            reduce
              ? { duration: 0 }
              : {
                  duration,
                  ease: 'easeInOut',
                  ...(loop && { repeat: Infinity, repeatType: 'loop' as const, repeatDelay: 0.3 }),
                }
          }
        />
      </svg>
    </Box>
  );
};

export default PulseLine;
