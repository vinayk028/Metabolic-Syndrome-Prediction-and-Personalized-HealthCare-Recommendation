/**
 * Resources Page
 * Educational materials, news, and references for metabolic syndrome
 */

import { useEffect } from 'react';
import {
    Box,
    Typography,
    Card,
    CardContent,
    CardMedia,
    Grid,
    Link,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Button,
    Chip,
    Skeleton,
} from '@mui/material';
import {
    ExpandMore as ExpandMoreIcon,
    Link as LinkIcon,
    LocalHospital as HospitalIcon,
    Newspaper as NewspaperIcon,
    FitnessCenter as FitnessIcon,
    MenuBook as BookIcon,
    Info as InfoIcon,
    OpenInNew as OpenInNewIcon,
    Refresh as RefreshIcon,
    AccessTime as TimeIcon,
    Science as ScienceIcon,
} from '@mui/icons-material';
import { useNewsStore } from '../../stores';
import './Resources.css';

// ==================== Helper Functions ====================

const getTimeAgo = (dateString: string): string => {
    const now = new Date();
    const date = new Date(dateString);
    const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
    return date.toLocaleDateString();
};

// ==================== Static Data ====================

const medicalResources = [
    {
        title: 'American Heart Association',
        description: 'Comprehensive information about metabolic syndrome and heart health',
        url: 'https://www.heart.org/en/health-topics/metabolic-syndrome',
    },
    {
        title: 'Mayo Clinic',
        description: 'Symptoms, causes, and treatment of metabolic syndrome',
        url: 'https://www.mayoclinic.org/diseases-conditions/metabolic-syndrome/symptoms-causes/syc-20351916',
    },
    {
        title: 'National Heart, Lung, and Blood Institute',
        description: 'In-depth medical research and guidelines',
        url: 'https://www.nhlbi.nih.gov/health-topics/metabolic-syndrome',
    },
    {
        title: 'World Health Organization',
        description: 'Global health data on noncommunicable diseases',
        url: 'https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases',
    },
];

const researchResources = [
    {
        title: 'PubMed',
        description: 'Latest research articles on metabolic syndrome',
        url: 'https://pubmed.ncbi.nlm.nih.gov/?term=metabolic+syndrome',
    },
    {
        title: 'American Diabetes Association',
        description: 'Research on metabolic syndrome and diabetes connection',
        url: 'https://diabetes.org/diabetes-risk/prediabetes/metabolic-syndrome',
    },
];

const lifestyleResources = [
    {
        title: 'DASH Diet',
        description: 'Dietary approaches to stop hypertension',
        url: 'https://www.nhlbi.nih.gov/health-topics/dash-eating-plan',
    },
    {
        title: 'Physical Activity Guidelines - CDC',
        description: 'Recommended physical activity for adults',
        url: 'https://www.cdc.gov/physicalactivity/basics/index.htm',
    },
    {
        title: 'Stress Management - Mayo Clinic',
        description: 'Techniques for managing stress effectively',
        url: 'https://www.mayoclinic.org/healthy-lifestyle/stress-management/in-depth/stress-management/art-20044289',
    },
];

const monitoringResources = [
    {
        title: 'Blood Pressure Monitoring',
        description: 'Guide to understanding blood pressure readings',
        url: 'https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings',
    },
    {
        title: 'Blood Glucose Monitoring - CDC',
        description: 'How to monitor and manage blood sugar levels',
        url: 'https://www.cdc.gov/diabetes/managing/managing-blood-sugar/bloodglucosemonitoring.html',
    },
];

const faqs = [
    {
        question: 'What causes metabolic syndrome?',
        answer: 'Metabolic syndrome is closely linked to overweight/obesity and inactivity. It\'s also linked to insulin resistance, which causes the body to have difficulty using insulin effectively. Genetic factors, aging, and hormonal changes can also contribute to developing metabolic syndrome.',
    },
    {
        question: 'Can metabolic syndrome be reversed?',
        answer: 'Yes, metabolic syndrome can often be reversed or managed through lifestyle changes. Weight loss, regular physical activity, healthy diet, stress management, and adequate sleep can significantly improve or eliminate metabolic syndrome conditions. In some cases, medication may also be prescribed.',
    },
    {
        question: 'How is metabolic syndrome diagnosed?',
        answer: 'Metabolic syndrome is diagnosed when a person has at least three of the five following conditions: large waist circumference, high triglycerides, low HDL cholesterol, high blood pressure, and high fasting blood sugar. A healthcare provider will perform blood tests and physical measurements.',
    },
    {
        question: 'What are the long-term risks of untreated metabolic syndrome?',
        answer: 'If left untreated, metabolic syndrome significantly increases the risk of developing type 2 diabetes (5x higher risk), cardiovascular disease (2x higher risk), stroke, and other serious health conditions. Early detection and management are crucial for prevention.',
    },
    {
        question: 'How often should I get screened for metabolic syndrome?',
        answer: 'Adults over 40, or those with risk factors like obesity, family history of diabetes, or sedentary lifestyle, should be screened annually. Younger adults with risk factors should discuss screening frequency with their healthcare provider.',
    },
];

const DEFAULT_NEWS_IMAGE = 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?w=400';

// ==================== Component ====================

const Resources = () => {
    const {
        resourceArticles: news,
        resourceLoading: newsLoading,
        resourceError: newsError,
        fetchResourceNews: fetchNews,
    } = useNewsStore();

    useEffect(() => {
        fetchNews();
    }, [fetchNews]);

    const handleImageError = (e: React.SyntheticEvent<HTMLImageElement>) => {
        e.currentTarget.src = DEFAULT_NEWS_IMAGE;
    };

    return (
        <Box className="resources-page">
            {/* Page Header */}
            <Box className="page-header">
                <Typography variant="h2" className="page-title">
                    Resources & References
                </Typography>
                <Typography variant="h6" className="page-subtitle">
                    Educational materials and trusted sources for metabolic syndrome information
                </Typography>
            </Box>

            {/* News Section */}
            <Card className="news-section-card">
                <CardContent>
                    <Box className="card-header news-header">
                        <Box className="news-title-section">
                            <NewspaperIcon className="card-icon news-icon" />
                            <Typography variant="h5" className="card-title">
                                Latest MetS News
                            </Typography>
                        </Box>
                        <Button
                            startIcon={<RefreshIcon />}
                            onClick={fetchNews}
                            disabled={newsLoading}
                            className="refresh-btn"
                            size="small"
                        >
                            Refresh
                        </Button>
                    </Box>

                    {newsLoading ? (
                        <Grid container spacing={3} className="news-grid">
                            {[1, 2, 3, 4, 5, 6].map((item) => (
                                <Grid size={{ xs: 12, sm: 6, md: 4 }} key={item}>
                                    <Card className="news-card-skeleton">
                                        <Skeleton variant="rectangular" height={160} />
                                        <CardContent>
                                            <Skeleton variant="text" width="40%" height={24} />
                                            <Skeleton variant="text" width="100%" height={28} />
                                            <Skeleton variant="text" width="100%" />
                                            <Skeleton variant="text" width="80%" />
                                        </CardContent>
                                    </Card>
                                </Grid>
                            ))}
                        </Grid>
                    ) : newsError && news.length === 0 ? (
                        <Box className="news-error">
                            <Typography variant="body1" color="textSecondary">
                                {newsError}
                            </Typography>
                            <Button
                                variant="outlined"
                                startIcon={<RefreshIcon />}
                                onClick={fetchNews}
                                sx={{ mt: 2 }}
                            >
                                Try Again
                            </Button>
                        </Box>
                    ) : (
                        <Grid container spacing={3} className="news-grid">
                            {news.map((article, index) => (
                                <Grid size={{ xs: 12, sm: 6, md: 4 }} key={index}>
                                    <Card className="news-card">
                                        <CardMedia
                                            component="img"
                                            height="160"
                                            image={article.image || DEFAULT_NEWS_IMAGE}
                                            alt={article.title}
                                            className="news-card-image"
                                            onError={handleImageError}
                                        />
                                        <CardContent className="news-card-content">
                                            <Box className="news-meta">
                                                <Chip
                                                    label={article.source}
                                                    size="small"
                                                    className="news-source-chip"
                                                />
                                                <Typography variant="caption" className="news-time">
                                                    <TimeIcon sx={{ fontSize: 14, mr: 0.5 }} />
                                                    {getTimeAgo(article.publishedAt)}
                                                </Typography>
                                            </Box>
                                            <Typography variant="h6" className="news-title">
                                                {article.title}
                                            </Typography>
                                            <Typography variant="body2" className="news-description">
                                                {article.description}
                                            </Typography>
                                            <Link
                                                href={article.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="news-link"
                                            >
                                                Read Full Article <OpenInNewIcon sx={{ fontSize: 16, ml: 0.5 }} />
                                            </Link>
                                        </CardContent>
                                    </Card>
                                </Grid>
                            ))}
                        </Grid>
                    )}
                </CardContent>
            </Card>

            {/* Resource Cards Grid */}
            <Grid container spacing={3}>
                <Grid size={{ xs: 12, md: 6 }}>
                    <Card className="resource-card">
                        <CardContent>
                            <Box className="card-header">
                                <HospitalIcon className="card-icon hospital-icon" />
                                <Typography variant="h5" className="card-title">
                                    Medical Organizations
                                </Typography>
                            </Box>
                            <List>
                                {medicalResources.map((resource, index) => (
                                    <ListItem key={index} className="resource-item">
                                        <ListItemIcon>
                                            <LinkIcon className="link-icon" />
                                        </ListItemIcon>
                                        <ListItemText
                                            primary={
                                                <Link href={resource.url} target="_blank" rel="noopener noreferrer" className="resource-link">
                                                    {resource.title}
                                                </Link>
                                            }
                                            secondary={resource.description}
                                        />
                                    </ListItem>
                                ))}
                            </List>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid size={{ xs: 12, md: 6 }}>
                    <Card className="resource-card">
                        <CardContent>
                            <Box className="card-header">
                                <ScienceIcon className="card-icon science-icon" />
                                <Typography variant="h5" className="card-title">
                                    Research & Studies
                                </Typography>
                            </Box>
                            <List>
                                {researchResources.map((resource, index) => (
                                    <ListItem key={index} className="resource-item">
                                        <ListItemIcon>
                                            <LinkIcon className="link-icon" />
                                        </ListItemIcon>
                                        <ListItemText
                                            primary={
                                                <Link href={resource.url} target="_blank" rel="noopener noreferrer" className="resource-link">
                                                    {resource.title}
                                                </Link>
                                            }
                                            secondary={resource.description}
                                        />
                                    </ListItem>
                                ))}
                            </List>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid size={{ xs: 12, md: 6 }}>
                    <Card className="resource-card">
                        <CardContent>
                            <Box className="card-header">
                                <FitnessIcon className="card-icon fitness-icon" />
                                <Typography variant="h5" className="card-title">
                                    Lifestyle Management
                                </Typography>
                            </Box>
                            <List>
                                {lifestyleResources.map((resource, index) => (
                                    <ListItem key={index} className="resource-item">
                                        <ListItemIcon>
                                            <LinkIcon className="link-icon" />
                                        </ListItemIcon>
                                        <ListItemText
                                            primary={
                                                <Link href={resource.url} target="_blank" rel="noopener noreferrer" className="resource-link">
                                                    {resource.title}
                                                </Link>
                                            }
                                            secondary={resource.description}
                                        />
                                    </ListItem>
                                ))}
                            </List>
                        </CardContent>
                    </Card>
                </Grid>

                <Grid size={{ xs: 12, md: 6 }}>
                    <Card className="resource-card">
                        <CardContent>
                            <Box className="card-header">
                                <BookIcon className="card-icon book-icon" />
                                <Typography variant="h5" className="card-title">
                                    Monitoring Tools
                                </Typography>
                            </Box>
                            <List>
                                {monitoringResources.map((resource, index) => (
                                    <ListItem key={index} className="resource-item">
                                        <ListItemIcon>
                                            <LinkIcon className="link-icon" />
                                        </ListItemIcon>
                                        <ListItemText
                                            primary={
                                                <Link href={resource.url} target="_blank" rel="noopener noreferrer" className="resource-link">
                                                    {resource.title}
                                                </Link>
                                            }
                                            secondary={resource.description}
                                        />
                                    </ListItem>
                                ))}
                            </List>
                        </CardContent>
                    </Card>
                </Grid>
            </Grid>

            {/* FAQ Section */}
            <Card className="faq-card">
                <CardContent>
                    <Box className="card-header centered">
                        <InfoIcon className="card-icon faq-icon" />
                        <Typography variant="h5" className="card-title">
                            Frequently Asked Questions
                        </Typography>
                    </Box>
                    <Box className="faq-container">
                        {faqs.map((faq, index) => (
                            <Accordion key={index} className="faq-accordion">
                                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                    <Typography className="faq-question">{faq.question}</Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <Typography className="faq-answer">{faq.answer}</Typography>
                                </AccordionDetails>
                            </Accordion>
                        ))}
                    </Box>
                </CardContent>
            </Card>

            {/* Disclaimer */}
            <Card className="disclaimer-card">
                <CardContent>
                    <Typography variant="h6" className="disclaimer-title">
                        Disclaimer
                    </Typography>
                    <Typography variant="body2" className="disclaimer-text">
                        This application is for informational and educational purposes only. It is not intended to
                        replace professional medical advice, diagnosis, or treatment. Always consult with a qualified 
                        healthcare provider for any health concerns.
                    </Typography>
                </CardContent>
            </Card>
        </Box>
    );
};

export default Resources;
