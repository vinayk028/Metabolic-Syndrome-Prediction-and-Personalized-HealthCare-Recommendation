/**
 * NotFound — shown for any address that doesn't match a page.
 * Same heading style as About / Resources, with clear ways back.
 */
import { Link, useLocation } from 'react-router-dom';
import { Button, Typography } from '@mui/material';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import HomeRoundedIcon from '@mui/icons-material/HomeRounded';
import TravelExploreRoundedIcon from '@mui/icons-material/TravelExploreRounded';
import './NotFound.css';

const NotFound = () => {
  const { pathname } = useLocation();
  return (
    <section className="nf" aria-labelledby="nf-title">
      <span className="nf__icon" aria-hidden>
        <TravelExploreRoundedIcon />
      </span>
      <span className="nf__eyebrow">Error 404</span>
      <Typography id="nf-title" variant="h2" component="h1" className="nf__title">
        Page not found
      </Typography>
      <Typography className="nf__sub">
        We couldn't find <code className="nf__path">{pathname}</code>. The link may be mistyped, or the page has moved.
      </Typography>
      <div className="nf__actions">
        <Button component={Link} to="/" variant="gradient" size="large" startIcon={<HomeRoundedIcon />}>
          Go to home
        </Button>
        <Button component={Link} to="/assessment" variant="outlined" size="large" endIcon={<ArrowForwardRoundedIcon />}>
          Start free assessment
        </Button>
      </div>
    </section>
  );
};

export default NotFound;
