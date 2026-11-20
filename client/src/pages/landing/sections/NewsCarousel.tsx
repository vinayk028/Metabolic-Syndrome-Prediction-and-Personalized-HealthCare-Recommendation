/**
 * NewsCarousel — swipeable (touch), scrollable (trackpad) and arrow-driven
 * (mouse/keyboard) news cards using native CSS scroll-snap.
 * Hidden entirely if the news service has nothing to show.
 */
import { useEffect, useRef, useState, type SyntheticEvent } from 'react';
import { Link } from 'react-router-dom';
import { Container, IconButton, Skeleton, Typography } from '@mui/material';
import ChevronLeftRoundedIcon from '@mui/icons-material/ChevronLeftRounded';
import ChevronRightRoundedIcon from '@mui/icons-material/ChevronRightRounded';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import OpenInNewRoundedIcon from '@mui/icons-material/OpenInNewRounded';
import { useNewsStore } from '../../../stores';
import { SectionHeader } from '../../../components/ui';
import DEFAULT_NEWS_IMAGE from '../../../assets/news.png';

const formatDate = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
};

const NewsCarousel = () => {
  const { slideshowArticles, slideshowLoading, fetchSlideshowNews, getHighQualityImage } = useNewsStore();
  const trackRef = useRef<HTMLUListElement>(null);
  const [edges, setEdges] = useState({ start: true, end: false });

  useEffect(() => {
    fetchSlideshowNews();
  }, [fetchSlideshowNews]);

  const updateEdges = () => {
    const el = trackRef.current;
    if (!el) return;
    setEdges({ start: el.scrollLeft <= 4, end: el.scrollLeft + el.clientWidth >= el.scrollWidth - 4 });
  };

  const scrollBy = (dir: 1 | -1) => {
    const el = trackRef.current;
    if (!el) return;
    el.scrollBy({ left: dir * el.clientWidth * 0.85, behavior: 'smooth' });
  };

  const onImgError = (e: SyntheticEvent<HTMLImageElement>) => {
    e.currentTarget.src = DEFAULT_NEWS_IMAGE;
  };

  if (!slideshowLoading && slideshowArticles.length === 0) return null;

  return (
    <section id="news" className="lp-section" aria-labelledby="lp-news-title">
      <Container maxWidth="lg">
        <div className="lp-news__head">
          <SectionHeader
            eyebrow="In the news"
            title={<span id="lp-news-title">Latest on metabolic health</span>}
            sx={{ mb: 0 }}
          />
          <div className="lp-news__arrows">
            <IconButton onClick={() => scrollBy(-1)} disabled={edges.start} aria-label="Previous stories">
              <ChevronLeftRoundedIcon />
            </IconButton>
            <IconButton onClick={() => scrollBy(1)} disabled={edges.end} aria-label="More stories">
              <ChevronRightRoundedIcon />
            </IconButton>
          </div>
        </div>

        <ul ref={trackRef} className="lp-news__track" onScroll={updateEdges} aria-busy={slideshowLoading}>
          {slideshowLoading
            ? Array.from({ length: 4 }).map((_, i) => (
                <li key={i} className="lp-news__card is-loading">
                  <Skeleton variant="rectangular" className="lp-news__img" />
                  <div className="lp-news__body">
                    <Skeleton width="40%" />
                    <Skeleton width="95%" />
                    <Skeleton width="70%" />
                  </div>
                </li>
              ))
            : slideshowArticles.map((a) => (
                <li key={a.url} className="lp-news__card">
                  <a href={a.url} target="_blank" rel="noopener noreferrer" className="lp-news__link">
                    <img
                      src={getHighQualityImage(a.image)}
                      alt=""
                      loading="lazy"
                      className="lp-news__img"
                      onError={onImgError}
                    />
                    <div className="lp-news__body">
                      <span className="lp-news__meta">
                        {a.source}
                        {formatDate(a.publishedAt) && ` · ${formatDate(a.publishedAt)}`}
                      </span>
                      <Typography component="h3" className="lp-news__title">
                        {a.title}
                      </Typography>
                      <span className="lp-news__read">
                        Read story <OpenInNewRoundedIcon aria-hidden />
                        <span className="visually-hidden"> (opens in a new tab)</span>
                      </span>
                    </div>
                  </a>
                </li>
              ))}
          {!slideshowLoading && (
            <li className="lp-news__card lp-news__card--more">
              <Link to="/resources" className="lp-news__link lp-news__more">
                <span className="lp-news__more-title">More news & resources</span>
                <span className="lp-news__more-cta">
                  Explore <ArrowForwardRoundedIcon aria-hidden />
                </span>
              </Link>
            </li>
          )}
        </ul>
      </Container>
    </section>
  );
};

export default NewsCarousel;
