/**
 * AnimatedReveal — fades/slides content in when it scrolls into view.
 * Automatically disabled for users who prefer reduced motion.
 */
import type { ReactNode } from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { motion as tokens } from '../../theme/tokens';

type Direction = 'up' | 'down' | 'left' | 'right' | 'none';

interface AnimatedRevealProps {
  children: ReactNode;
  direction?: Direction;
  /** Seconds before the animation starts (use for staggering siblings). */
  delay?: number;
  /** Travel distance in px. */
  distance?: number;
  once?: boolean;
  className?: string;
  as?: 'div' | 'section' | 'li' | 'article';
}

const offset = (direction: Direction, d: number) => {
  switch (direction) {
    case 'up':
      return { y: d };
    case 'down':
      return { y: -d };
    case 'left':
      return { x: d };
    case 'right':
      return { x: -d };
    default:
      return {};
  }
};

const AnimatedReveal = ({
  children,
  direction = 'up',
  delay = 0,
  distance = 24,
  once = true,
  className,
  as = 'div',
}: AnimatedRevealProps) => {
  const reduce = useReducedMotion();
  const MotionTag = motion[as];

  if (reduce) {
    return <MotionTag className={className}>{children}</MotionTag>;
  }

  return (
    <MotionTag
      className={className}
      initial={{ opacity: 0, ...offset(direction, distance) }}
      whileInView={{ opacity: 1, x: 0, y: 0 }}
      viewport={{ once, margin: '0px 0px -10% 0px' }}
      transition={{ duration: tokens.duration.slow, delay, ease: tokens.ease.emphasized }}
    >
      {children}
    </MotionTag>
  );
};

export default AnimatedReveal;
