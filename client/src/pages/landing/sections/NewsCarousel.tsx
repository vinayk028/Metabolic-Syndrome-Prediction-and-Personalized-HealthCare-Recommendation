/**
 * NewsCarousel — a streaming-app style row of news cards.
 * Swipe on touch, scroll with a trackpad, or use the round arrows that sit
 * over the left/right edges of the images (shown on hover, hidden at the ends).
 * Hidden entirely if the news service has nothing to show.
 */
import { useEffect, useRef, useState, type SyntheticEvent } from 'react';
import { Link } from 'react-router-dom';
import { Button, Container, Skeleton, Typography } from '@mui/material';
import ChevronLeftRoundedIcon from '@mui/icons-material/ChevronLeftRounded';
import ChevronRightRoundedIcon from '@mui/icons-material/ChevronRightRounded';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import OpenInNewRoundedIcon from '@mui/icons-material/OpenInNewRounded';
import { useNewsStore } from '../../../stores';
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

  // Re-check the ends whenever the row resizes (also runs once on mount)
  useEffect(() => {
    const el = trackRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => updateEdges());
    ro.observe(el);
    return () => ro.disconnect();
  }, [slideshowLoading]);

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
    <section id="news" className="lp-section lp-news" aria-labelledby="lp-news-title">
      <Container maxWidth="xl">
        <div className="lp-news__head">
          <div>
            <span className="lp-kicker">In the news</span>
            <Typography id="lp-news-title" variant="h2" className="lp-h2">
              Latest on <em className="vb-accent-text">metabolic health</em>
            </Typography>
          </div>
          <Button component={Link} to="/resources" endIcon={<ArrowForwardRoundedIcon />} className="lp-news__all">
            All news
          </Button>
        </div>

        <div
          className={`lp-news__rail${edges.start ? ' at-start' : ''}${edges.end ? ' at-end' : ''}`}
          aria-roledescription="carousel"
        >
          <button
            type="button"
            className="lp-news__arrow lp-news__arrow--prev"
            onClick={() => scrollBy(-1)}
            aria-label="Previous stories"
            tabIndex={edges.start ? -1 : 0}
            aria-hidden={edges.start}
          >
            <ChevronLeftRoundedIcon />
          </button>

          <ul ref={trackRef} className="lp-news__track" onScroll={updateEdges} aria-busy={slideshowLoading}>
            {slideshowLoading
              ? Array.from({ length: 5 }).map((_, i) => (
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
                      <span className="lp-news__media">
                        <img src={getHighQualityImage(a.image)} alt="" loading="lazy" className="lp-news__img" onError={onImgError} />
                      </span>
                      <div className="lp-news__body">
                        <span className="lp-news__meta">
                          <span className="lp-news__source">{a.source}</span>
                          {formatDate(a.publishedAt) && <span> · {formatDate(a.publishedAt)}</span>}
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
                  <span className="lp-news__more-title">More news &amp; resources</span>
                  <span className="lp-news__more-cta">
                    Explore <ArrowForwardRoundedIcon aria-hidden />
                  </span>
                </Link>
              </li>
            )}
          </ul>

          <button
            type="button"
            className="lp-news__arrow lp-news__arrow--next"
            onClick={() => scrollBy(1)}
            aria-label="More stories"
            tabIndex={edges.end ? -1 : 0}
            aria-hidden={edges.end}
          >
            <ChevronRightRoundedIcon />
          </button>
        </div>
      </Container>
    </section>
  );
};

export default NewsCarousel;
