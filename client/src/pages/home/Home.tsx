import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  Skeleton,
} from '@mui/material';
import {
  Assessment as AssessmentIcon,
  Favorite as HeartIcon,
  LocalHospital as HospitalIcon,
  Psychology as BrainIcon,
  FitnessCenter as FitnessIcon,
  Restaurant as DietIcon,
  Newspaper as NewspaperIcon,
  ArrowForward as ArrowForwardIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
} from '@mui/icons-material';
import { useNewsStore } from '../../stores';
import HERO_IMAGE from '../../assets/hero.jpg';
import INFO_IMAGE from '../../assets/info.jpg';
import CTA_IMAGE from '../../assets/CTA.jpg';
import DEFAULT_NEWS_IMAGE from '../../assets/news.png';
import './Home.css';

const features = [
  {
    icon: <AssessmentIcon />,
    title: 'Risk Assessment',
    description: 'Get an accurate prediction of your metabolic syndrome risk based on your health data.',
    colorClass: 'feature-teal',
  },
  {
    icon: <BrainIcon />,
    title: 'AI-Powered Analysis',
    description: 'Advanced algorithms analyze your health metrics for precise risk evaluation.',
    colorClass: 'feature-purple',
  },
  {
    icon: <DietIcon />,
    title: 'Diet Recommendations',
    description: 'Personalized diet plans tailored to your risk level and health profile.',
    colorClass: 'feature-green',
  },
  {
    icon: <FitnessIcon />,
    title: 'Exercise Plans',
    description: 'Custom exercise routines designed to help manage metabolic health.',
    colorClass: 'feature-orange',
  },
  {
    icon: <HospitalIcon />,
    title: 'Health Insights',
    description: 'Comprehensive health insights and severity analysis for better understanding.',
    colorClass: 'feature-blue',
  },
  {
    icon: <HeartIcon />,
    title: 'Wellness Journey',
    description: 'Track your progress and download personalized health plans.',
    colorClass: 'feature-rose',
  },
];

const Home = () => {
  const navigate = useNavigate();
  const { slideshowArticles, slideshowLoading, fetchSlideshowNews, getHighQualityImage } = useNewsStore();
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isPaused, setIsPaused] = useState(false);
  const slideInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  // Total slides = news articles + 1 (View More slide)
  const totalSlides = slideshowArticles.length + 1;

  const goToSlide = useCallback((index: number) => {
    setCurrentSlide(index);
  }, []);

  const nextSlide = useCallback(() => {
    setCurrentSlide((prev) => (prev + 1) % totalSlides);
  }, [totalSlides]);

  const prevSlide = useCallback(() => {
    setCurrentSlide((prev) => (prev - 1 + totalSlides) % totalSlides);
  }, [totalSlides]);

  // Fetch slideshow news on mount
  useEffect(() => {
    fetchSlideshowNews();
  }, [fetchSlideshowNews]);

  // Auto-slide every 4 seconds
  useEffect(() => {
    if (slideshowArticles.length === 0 || isPaused) return;

    slideInterval.current = setInterval(() => {
      nextSlide();
    }, 4000);

    return () => {
      if (slideInterval.current) clearInterval(slideInterval.current);
    };
  }, [slideshowArticles.length, isPaused, nextSlide]);

  const handleImageError = (e: React.SyntheticEvent<HTMLImageElement>) => {
    e.currentTarget.src = DEFAULT_NEWS_IMAGE;
  };

  return (
    <Box className="home-page">
      <Box className="hero-section">
        <Box className="hero-content">
          <Typography variant="h1" className="hero-title">
            Metabolic Syndrome
            <span className="highlight"> Predictor</span>
          </Typography>
          <Typography variant="h5" className="hero-subtitle">
            Assess your risk, understand your health, and get personalized recommendations
            for a healthier lifestyle.
          </Typography>
          <Box className="hero-buttons">
            <Button
              variant="contained"
              size="large"
              className="primary-button"
              onClick={() => navigate('/assessment')}
              startIcon={<AssessmentIcon />}
            >
              Start Assessment
            </Button>
            <Button
              variant="outlined"
              size="large"
              className="secondary-button"
              onClick={() => navigate('/about')}
            >
              Learn More
            </Button>
          </Box>
        </Box>
        <Box className="hero-illustration">
          <Box className="hero-image-container">
            <img 
              src={HERO_IMAGE} 
              alt="Healthcare professional with stethoscope" 
              className="hero-image"
            />
            <Box className="hero-image-overlay"></Box>
          </Box>
          <Box className="pulse-circle"></Box>
          <HeartIcon className="hero-heart-icon" />
        </Box>
      </Box>

      <Box className="info-section">
        <Card className="info-card">
          <CardContent>
            <Box className="info-content-wrapper">
              <Box className="info-image-container">
                <img 
                  src={INFO_IMAGE} 
                  alt="Health checkup and monitoring" 
                  className="info-image"
                />
              </Box>
              <Box className="info-text-content">
                <Typography variant="h4" className="info-title">
                  What is Metabolic Syndrome?
                </Typography>
                <Typography variant="body1" className="info-text">
                  Metabolic syndrome is a cluster of conditions that occur together, increasing 
                  your risk of heart disease, stroke, and type 2 diabetes. These conditions include 
                  increased blood pressure, high blood sugar, excess body fat around the waist, 
                  and abnormal cholesterol or triglyceride levels.
                </Typography>
              </Box>
            </Box>
            <Box className="stats-container">
              <Box className="stat-item">
                <Typography variant="h3" className="stat-number">1 in 3</Typography>
                <Typography variant="body2" className="stat-label">Adults Affected</Typography>
              </Box>
              <Box className="stat-item">
                <Typography variant="h3" className="stat-number">5x</Typography>
                <Typography variant="body2" className="stat-label">Higher Diabetes Risk</Typography>
              </Box>
              <Box className="stat-item">
                <Typography variant="h3" className="stat-number">2x</Typography>
                <Typography variant="body2" className="stat-label">Higher Heart Disease Risk</Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Box>

      {/* News Slideshow Section */}
      {slideshowLoading ? (
        <Box className="news-slideshow-section">
          <Card className="news-slideshow-card">
            <Skeleton variant="rectangular" height={320} sx={{ borderRadius: '24px' }} />
          </Card>
        </Box>
      ) : slideshowArticles.length > 0 ? (
        <Box className="news-slideshow-section">
          <Typography variant="h3" className="section-title">
            <NewspaperIcon sx={{ fontSize: '2rem', mr: 1, verticalAlign: 'middle', color: '#00b2a7' }} />
            MetS News
          </Typography>
          <Card
            className="news-slideshow-card"
            onMouseEnter={() => setIsPaused(true)}
            onMouseLeave={() => setIsPaused(false)}
          >
            <Box className="news-slideshow-track" style={{ transform: `translateX(-${currentSlide * 100}%)` }}>
              {slideshowArticles.map((article, index) => (
                <Box className="news-slide" key={index}>
                  <img
                    src={getHighQualityImage(article.image)}
                    alt={article.title}
                    className="news-slide-image"
                    onError={handleImageError}
                  />
                  <Box className="news-slide-overlay">
                    <Box className="news-slide-content">
                      <Typography variant="overline" className="news-slide-source">
                        {article.source}
                      </Typography>
                      <Typography variant="h5" className="news-slide-title">
                        {article.title}
                      </Typography>
                      <Typography variant="body2" className="news-slide-description">
                        {article.description}
                      </Typography>
                    </Box>
                  </Box>
                </Box>
              ))}
              {/* View More Slide */}
              <Box className="news-slide news-slide-viewmore">
                <Box className="news-slide-viewmore-content">
                  <NewspaperIcon sx={{ fontSize: '4rem', color: '#00b2a7', mb: 2 }} />
                  <Typography variant="h4" className="news-slide-viewmore-title">
                    Want to read more?
                  </Typography>
                  <Typography variant="body1" className="news-slide-viewmore-text">
                    Explore all the latest news and resources about metabolic syndrome.
                  </Typography>
                  <Button
                    variant="contained"
                    size="large"
                    className="news-slide-viewmore-btn"
                    endIcon={<ArrowForwardIcon />}
                    onClick={() => navigate('/resources')}
                  >
                    View More News
                  </Button>
                </Box>
              </Box>
            </Box>

            {/* Navigation Arrows */}
            <button className="news-slide-arrow news-slide-arrow-left" onClick={prevSlide} aria-label="Previous slide">
              <ChevronLeftIcon />
            </button>
            <button className="news-slide-arrow news-slide-arrow-right" onClick={nextSlide} aria-label="Next slide">
              <ChevronRightIcon />
            </button>

            {/* Dots Indicator */}
            <Box className="news-slide-dots">
              {Array.from({ length: totalSlides }).map((_, index) => (
                <span
                  key={index}
                  className={`news-slide-dot ${currentSlide === index ? 'active' : ''}`}
                  onClick={() => goToSlide(index)}
                />
              ))}
            </Box>
          </Card>
        </Box>
      ) : null}

      <Box className="features-section">
        <Typography variant="h3" className="section-title">
          How We Help You
        </Typography>
        <Grid container spacing={3}>
          {features.map((feature, index) => (
            <Grid size={{ xs: 12, sm: 6, md: 4 }} key={index}>
              <Card className="feature-card">
                <CardContent>
                  <Box className={`feature-icon ${feature.colorClass}`}>{feature.icon}</Box>
                  <Typography variant="h6" className="feature-title">
                    {feature.title}
                  </Typography>
                  <Typography variant="body2" className="feature-description">
                    {feature.description}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      <Box className="cta-section">
        <Card className="cta-card">
          <CardContent className="cta-content-wrapper">
            <Box className="cta-image-container">
              <img 
                src={CTA_IMAGE} 
                alt="Healthy lifestyle and fitness" 
                className="cta-image"
              />
            </Box>
            <Box className="cta-text-content">
              <Typography variant="h4" className="cta-title">
                Ready to Take Control of Your Health?
              </Typography>
              <Typography variant="body1" className="cta-text">
                Start your assessment now and receive personalized recommendations 
                based on your health profile.
              </Typography>
              <Button
                variant="contained"
                size="large"
                className="cta-button"
                onClick={() => navigate('/assessment')}
              >
                Begin Your Assessment
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};

export default Home;
