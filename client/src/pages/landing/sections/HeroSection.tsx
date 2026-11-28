/**
 * Hero — light, softly tinted stage: headline and CTAs on the
 * left, the interactive marker ring on the right (stacked on narrow screens).
 */
import { Link } from 'react-router-dom';
import { Button, Container, Typography } from '@mui/material';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import SpaceDashboardRoundedIcon from '@mui/icons-material/SpaceDashboardRounded';
import { motion, useReducedMotion } from 'framer-motion';
import { useAuthStore } from '../../../stores';
import HeroRing from './HeroRing';

const HeroSection = () => {
  const reduce = useReducedMotion();
  const { user, isAuthenticated } = useAuthStore();
  const loggedIn = isAuthenticated && !!user;

  const fadeUp = (delay: number) =>
    reduce
      ? {}
      : {
          initial: { opacity: 0, y: 18 },
          animate: { opacity: 1, y: 0 },
          transition: { delay, duration: 0.7, ease: [0.2, 0, 0, 1] as const },
        };

  return (
    <>
      <section className="lp-hero" aria-labelledby="lp-hero-title">
        <div className="lp-hero__bg" aria-hidden />

        <Container maxWidth="xl" className="lp-hero__inner">
          <div className="lp-hero__copy">
            {loggedIn && (
              <motion.span className="lp-hero__badge lp-hero__welcome" {...fadeUp(0)}>
                <span className="lp-hero__badge-dot" />
                Welcome back, {user?.firstName}
              </motion.span>
            )}

          <motion.div {...fadeUp(0.08)}>
            <Typography id="lp-hero-title" variant="display" component="h1" className="lp-hero__title">
              Catch metabolic syndrome <em className="vb-accent-text">before it catches you.</em>
            </Typography>
          </motion.div>

          <motion.div {...fadeUp(0.16)}>
            <Typography variant="lead" className="lp-hero__lead">
              Enter five numbers from a routine check-up. Get your risk, how severe it is, and a personal plan to bring
              everything back into balance.
            </Typography>
          </motion.div>

          <motion.div className="lp-hero__actions" {...fadeUp(0.24)}>
            <Button component={Link} to="/assessment" variant="gradient" size="large" endIcon={<ArrowForwardRoundedIcon />}>
              {loggedIn ? 'Start a new assessment' : 'Start free assessment'}
            </Button>
            {loggedIn ? (
              <Button component={Link} to="/dashboard" size="large" className="lp-btn-ghost" startIcon={<SpaceDashboardRoundedIcon />}>
                Open dashboard
              </Button>
            ) : (
              <Button component="a" href="#how-it-works" size="large" className="lp-btn-ghost">
                See how it works
              </Button>
            )}
          </motion.div>

          </div>

          <motion.div
            className="lp-hero__visual"
            initial={reduce ? false : { opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.8, ease: [0.2, 0, 0, 1] }}
          >
            <HeroRing />
          </motion.div>
        </Container>
      </section>
    </>
  );
};

export default HeroSection;
