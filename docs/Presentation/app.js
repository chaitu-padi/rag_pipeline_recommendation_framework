// RAG Pipeline Optimization System - Interactive Presentation Controller

class PresentationController {
    constructor() {
        this.init();
    }

    init() {
        console.log('Initializing RAG Pipeline Optimization Presentation...');
        this.setupEventListeners();
        this.setupAnimations();
        this.setupInteractiveElements();
        this.initializeProgressIndicators();
    }

    setupEventListeners() {
        // Download functionality
        document.getElementById('downloadPDF')?.addEventListener('click', this.handlePDFDownload.bind(this));
        document.getElementById('downloadHTML')?.addEventListener('click', this.handleHTMLDownload.bind(this));
        document.getElementById('printPresentation')?.addEventListener('click', this.handlePrint.bind(this));

        // CTA buttons
        document.getElementById('approveProject')?.addEventListener('click', this.handleProjectApproval.bind(this));
        document.getElementById('requestDetails')?.addEventListener('click', this.handleDetailsRequest.bind(this));

        // Modal functionality
        document.getElementById('modalClose')?.addEventListener('click', this.closeModal.bind(this));
        document.querySelector('.modal-backdrop')?.addEventListener('click', this.closeModal.bind(this));

        // Section interactions
        this.setupSectionInteractions();

        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));

        // Scroll animations
        this.setupScrollAnimations();
    }

    setupSectionInteractions() {
        // Interactive sections for detailed information
        document.querySelectorAll('.presentation-section').forEach(section => {
            section.addEventListener('click', (e) => {
                if (!e.target.closest('.action-buttons') && !e.target.closest('.cta-button')) {
                    this.handleSectionClick(section);
                }
            });
        });

        // Feature items interactions
        document.querySelectorAll('.feature-item').forEach(item => {
            item.addEventListener('mouseenter', this.handleFeatureHover.bind(this));
            item.addEventListener('click', this.handleFeatureClick.bind(this));
        });

        // Process steps interactions
        document.querySelectorAll('.step-item').forEach(step => {
            step.addEventListener('click', this.handleStepClick.bind(this));
        });

        // Metric cards interactions
        document.querySelectorAll('.metric-card').forEach(card => {
            card.addEventListener('click', this.handleMetricClick.bind(this));
        });
    }

    setupAnimations() {
        // Counter animations for metrics
        this.animateCounters();
        
        // Entrance animations
        this.animateOnScroll();
        
        // Interactive hover effects
        this.setupHoverEffects();
    }

    setupInteractiveElements() {
        // Add tooltips and interactive states
        this.addTooltips();
        
        // Setup progress indicators
        this.updateProgressIndicators();
    }

    initializeProgressIndicators() {
        // Add visual progress indicators for the presentation flow
        const sections = document.querySelectorAll('.presentation-section');
        sections.forEach((section, index) => {
            section.style.animationDelay = `${index * 0.1}s`;
        });
    }

    // Download Handlers
    async handlePDFDownload() {
        this.showToast('Generating PDF presentation...', 'info');
        
        try {
            const { jsPDF } = window.jspdf;
            const pdf = new jsPDF('p', 'mm', 'a4');
            
            // Add title page
            pdf.setFontSize(24);
            pdf.setFont(undefined, 'bold');
            pdf.text('RAG Pipeline Optimization System', 20, 30);
            
            pdf.setFontSize(16);
            pdf.setFont(undefined, 'normal');
            pdf.text('Preventing AI Investment Failures Through Data-Driven Automation', 20, 45);
            
            // Add executive summary
            pdf.setFontSize(14);
            pdf.setFont(undefined, 'bold');
            pdf.text('Executive Summary', 20, 65);
            
            pdf.setFontSize(12);
            pdf.setFont(undefined, 'normal');
            const summaryText = [
                '• 60% of RAG projects fail to reach production',
                '• Our AI-powered solution reduces failure rate to 20%',
                '• Investment protection of $300K-$400K per $1M invested',
                '• 6-month implementation timeline with proven technology stack',
                '• First-to-market advantage in automated RAG optimization'
            ];
            
            summaryText.forEach((line, index) => {
                pdf.text(line, 20, 80 + (index * 8));
            });

            // Add key benefits section
            pdf.addPage();
            pdf.setFontSize(16);
            pdf.setFont(undefined, 'bold');
            pdf.text('Key Benefits & ROI Analysis', 20, 30);
            
            const benefits = [
                'Performance Optimization: 25-40% improvement in accuracy',
                'Cost Reduction: 60-80% operational expense savings', 
                'Rapid Deployment: 90% faster time-to-production',
                'Risk Mitigation: 50% reduction in project failure rate',
                'Investment Protection: Proven ROI within 6 months'
            ];
            
            pdf.setFontSize(12);
            pdf.setFont(undefined, 'normal');
            benefits.forEach((benefit, index) => {
                pdf.text(`• ${benefit}`, 20, 50 + (index * 10));
            });

            // Add implementation details
            pdf.addPage();
            pdf.setFontSize(16);
            pdf.setFont(undefined, 'bold');
            pdf.text('Implementation Plan', 20, 30);
            
            pdf.setFontSize(12);
            pdf.setFont(undefined, 'normal');
            const implementation = [
                'Timeline: 6 months to production-ready system',
                'Resources: 4-6 experienced engineers + cloud infrastructure',
                'Phase 1 (Months 1-2): Core system development',
                'Phase 2 (Months 2-3): DASK integration & optimization',
                'Phase 3 (Months 3-4): UI development & testing',
                'Phase 4 (Months 4-6): Production deployment & monitoring'
            ];
            
            implementation.forEach((item, index) => {
                pdf.text(`• ${item}`, 20, 50 + (index * 10));
            });

            // Save the PDF
            pdf.save('RAG-Pipeline-Optimization-Presentation.pdf');
            this.showToast('PDF downloaded successfully!', 'success');
            
        } catch (error) {
            console.error('PDF generation error:', error);
            this.showToast('Error generating PDF. Please try again.', 'error');
        }
    }

    handleHTMLDownload() {
        this.showToast('Preparing HTML package...', 'info');
        
        try {
            // Create a complete HTML package
            const htmlContent = this.generateCompleteHTML();
            const blob = new Blob([htmlContent], { type: 'text/html' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = 'RAG-Pipeline-Optimization-Presentation.html';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            this.showToast('HTML package downloaded successfully!', 'success');
        } catch (error) {
            console.error('HTML download error:', error);
            this.showToast('Error downloading HTML. Please try again.', 'error');
        }
    }

    generateCompleteHTML() {
        const currentHTML = document.documentElement.outerHTML;
        // Inline CSS and make it standalone
        const cssContent = Array.from(document.styleSheets)
            .map(sheet => {
                try {
                    return Array.from(sheet.cssRules).map(rule => rule.cssText).join('\n');
                } catch (e) {
                    return '';
                }
            }).join('\n');
            
        return currentHTML.replace(
            '<link rel="stylesheet" href="style.css">',
            `<style>${cssContent}</style>`
        );
    }

    handlePrint() {
        this.showToast('Opening print dialog...', 'info');
        
        // Hide interactive elements for printing
        const downloadControls = document.querySelector('.download-controls');
        if (downloadControls) {
            downloadControls.style.display = 'none';
        }
        
        setTimeout(() => {
            window.print();
            
            // Restore elements after printing
            setTimeout(() => {
                if (downloadControls) {
                    downloadControls.style.display = 'flex';
                }
            }, 1000);
        }, 500);
    }

    // CTA Handlers
    handleProjectApproval() {
        this.showModal('Project Approval Confirmation', this.getApprovalContent());
        
        // Track interaction
        console.log('Project approval initiated');
    }

    handleDetailsRequest() {
        this.showModal('Detailed Proposal Request', this.getDetailsContent());
        
        // Track interaction
        console.log('Detailed proposal requested');
    }

    getApprovalContent() {
        return `
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="font-size: 48px; margin-bottom: 16px;">🎉</div>
                <h4 style="color: var(--color-success-bright); margin-bottom: 16px;">
                    RAG Pipeline Optimization System Approved!
                </h4>
            </div>
            
            <div style="background: var(--color-bg-3); padding: 16px; border-radius: 8px; margin-bottom: 20px;">
                <h5 style="color: var(--color-primary-deep); margin-bottom: 12px;">Immediate Next Steps:</h5>
                <ul style="margin: 0; padding-left: 16px;">
                    <li>Technical kickoff meeting scheduled within 48 hours</li>
                    <li>Resource allocation and budget approval process initiated</li>
                    <li>Infrastructure setup and environment provisioning begins</li>
                    <li>Phase 1 development sprint planning and team assignment</li>
                </ul>
            </div>
            
            <div style="background: var(--color-bg-1); padding: 16px; border-radius: 8px; margin-bottom: 20px;">
                <h5 style="color: var(--color-primary-deep); margin-bottom: 12px;">Timeline & Milestones:</h5>
                <p style="margin: 0; font-size: 14px;">
                    <strong>Week 1-2:</strong> Team assembly and architecture finalization<br>
                    <strong>Month 1:</strong> Core development begins with DASK integration<br>
                    <strong>Month 3:</strong> Alpha testing with internal datasets<br>
                    <strong>Month 6:</strong> Production-ready deployment
                </p>
            </div>
            
            <div style="background: var(--color-bg-2); padding: 16px; border-radius: 8px;">
                <h5 style="color: var(--color-primary-deep); margin-bottom: 12px;">Expected Outcomes:</h5>
                <p style="margin: 0; font-size: 14px;">
                    • <strong>80%+ RAG project success rate</strong> (vs current 40%)<br>
                    • <strong>$300K-400K investment protection</strong> per $1M deployed<br>
                    • <strong>First-to-market advantage</strong> in automated RAG optimization<br>
                    • <strong>Scalable platform</strong> for multiple business units
                </p>
            </div>
            
            <div style="text-align: center; margin-top: 24px; padding: 12px; background: var(--color-bg-5); border-radius: 8px;">
                <strong>Project Lead Contact:</strong> Technical team will reach out within 24 hours
            </div>
        `;
    }

    getDetailsContent() {
        return `
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="font-size: 48px; margin-bottom: 16px;">📋</div>
                <h4 style="color: var(--color-primary-deep); margin-bottom: 16px;">
                    Comprehensive Technical Proposal
                </h4>
            </div>
            
            <p style="margin-bottom: 20px;">
                A detailed 50+ page technical proposal will be prepared and delivered within 
                <strong>5 business days</strong>, including comprehensive documentation across all aspects:
            </p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px;">
                <div style="background: var(--color-bg-1); padding: 16px; border-radius: 8px;">
                    <h5 style="color: var(--color-primary-deep); margin-bottom: 12px;">📐 Technical Specifications</h5>
                    <ul style="font-size: 13px; margin: 0; padding-left: 16px;">
                        <li>Detailed system architecture diagrams</li>
                        <li>Complete API documentation and schemas</li>
                        <li>Database models and data flow diagrams</li>
                        <li>Security protocols and compliance framework</li>
                        <li>Performance benchmarking methodology</li>
                        <li>Scalability and load testing procedures</li>
                    </ul>
                </div>
                <div style="background: var(--color-bg-2); padding: 16px; border-radius: 8px;">
                    <h5 style="color: var(--color-primary-deep); margin-bottom: 12px;">💼 Business Analysis</h5>
                    <ul style="font-size: 13px; margin: 0; padding-left: 16px;">
                        <li>Detailed ROI calculations and projections</li>
                        <li>Comprehensive risk assessment matrix</li>
                        <li>Market analysis and competitive positioning</li>
                        <li>Implementation timeline with milestones</li>
                        <li>Resource allocation and budget breakdown</li>
                        <li>Success metrics and KPI definitions</li>
                    </ul>
                </div>
            </div>
            
            <div style="background: var(--color-bg-3); padding: 16px; border-radius: 8px; margin-bottom: 20px;">
                <h5 style="color: var(--color-primary-deep); margin-bottom: 12px;">🎯 Deliverables Included:</h5>
                <ul style="margin: 0; padding-left: 16px; font-size: 14px;">
                    <li><strong>Executive Summary</strong> (5 pages) - Strategic overview and business case</li>
                    <li><strong>Technical Deep-dive</strong> (25 pages) - Architecture, algorithms, and implementation</li>
                    <li><strong>Financial Models</strong> (10 pages) - ROI analysis, cost projections, and budget</li>
                    <li><strong>Implementation Roadmap</strong> (10 pages) - Timeline, resources, and milestones</li>
                    <li><strong>Risk Assessment</strong> (5 pages) - Technical and business risk mitigation</li>
                </ul>
            </div>
            
            <div style="background: var(--color-bg-5); padding: 16px; border-radius: 8px;">
                <h5 style="color: var(--color-primary-deep); margin-bottom: 12px;">🎁 Additional Resources:</h5>
                <p style="margin: 0; font-size: 14px;">
                    • <strong>Live Demo Session</strong> - Interactive proof-of-concept demonstration<br>
                    • <strong>Reference Implementations</strong> - Sample code and integration examples<br>
                    • <strong>Expert Consultation</strong> - Direct access to technical architects<br>
                    • <strong>Competitive Analysis</strong> - Positioning against alternative solutions
                </p>
            </div>
            
            <div style="text-align: center; margin-top: 24px; padding: 12px; background: linear-gradient(135deg, #1e40af 0%, #0891b2 100%); color: white; border-radius: 8px;">
                <strong>Proposal Delivery Timeline:</strong> Complete documentation within 5 business days
            </div>
        `;
    }

    // Section Interaction Handlers
    handleSectionClick(section) {
        const sectionClass = section.className;
        let title = 'Section Details';
        let content = 'Additional information about this section.';
        
        if (sectionClass.includes('problem-section')) {
            title = 'RAG Implementation Challenges';
            content = this.getProblemDetails();
        } else if (sectionClass.includes('solution-section')) {
            title = 'AI-Powered Optimization Solution';
            content = this.getSolutionDetails();
        } else if (sectionClass.includes('process-section')) {
            title = 'End-to-End Process Flow';
            content = this.getProcessDetails();
        } else if (sectionClass.includes('roi-section')) {
            title = 'Investment Protection & Returns';
            content = this.getROIDetails();
        } else if (sectionClass.includes('architecture-section')) {
            title = 'Technical Architecture Deep-Dive';
            content = this.getArchitectureDetails();
        }
        
        this.showModal(title, content);
    }

    getProblemDetails() {
        return `
            <h5 style="color: var(--color-warning-red);">Current RAG Implementation Challenges</h5>
            <div style="background: var(--color-bg-4); padding: 16px; border-radius: 8px; margin: 16px 0;">
                <h6>Statistical Reality:</h6>
                <ul>
                    <li><strong>60% Failure Rate:</strong> 6 out of 10 RAG projects never reach production</li>
                    <li><strong>Average Loss:</strong> $200K-500K per failed implementation</li>
                    <li><strong>Time Waste:</strong> 3-6 months of development time lost</li>
                    <li><strong>Resource Drain:</strong> 15-20 engineer-months typically wasted</li>
                </ul>
            </div>
            <h6>Root Causes:</h6>
            <ul>
                <li><strong>Manual Configuration:</strong> Trial-and-error approaches without data-driven optimization</li>
                <li><strong>Lack of Benchmarks:</strong> No systematic way to evaluate configuration choices</li>
                <li><strong>Performance Unpredictability:</strong> Unable to forecast system behavior on real data</li>
                <li><strong>Cost Spiraling:</strong> Compute and API costs escalate without optimization</li>
            </ul>
        `;
    }

    getSolutionDetails() {
        return `
            <h5 style="color: var(--color-success-bright);">AI-Powered RAG Optimization</h5>
            <div style="background: var(--color-bg-3); padding: 16px; border-radius: 8px; margin: 16px 0;">
                <h6>Core Innovation:</h6>
                <p>First system to combine external benchmark data with dynamic user evaluation using advanced ML algorithms.</p>
            </div>
            <h6>Technical Components:</h6>
            <ul>
                <li><strong>Hierarchical Multi-Armed Bandit:</strong> Optimizes configuration choices through intelligent exploration</li>
                <li><strong>Bayesian Optimization:</strong> Provides probabilistic performance predictions with confidence intervals</li>
                <li><strong>External Benchmark Integration:</strong> Leverages 1000+ proven RAG implementations</li>
                <li><strong>Dynamic Evaluation Engine:</strong> Real-time testing on user-specific data</li>
            </ul>
            <h6>Competitive Advantages:</h6>
            <ul>
                <li>Only solution providing automated, data-driven RAG optimization</li>
                <li>Reduces configuration time from weeks to minutes</li>
                <li>Proven performance improvements across diverse use cases</li>
                <li>Scalable architecture supporting enterprise deployments</li>
            </ul>
        `;
    }

    getProcessDetails() {
        return `
            <h5>Complete Process Flow Breakdown</h5>
            <div style="background: var(--color-bg-1); padding: 16px; border-radius: 8px; margin: 16px 0;">
                <h6>Process Timeline:</h6>
                <p><strong>Total Time:</strong> 15 minutes vs. traditional 2-4 weeks</p>
            </div>
            <h6>Step-by-Step Details:</h6>
            <ol style="padding-left: 20px;">
                <li><strong>User Requirements:</strong> Simple form specifying performance priorities and constraints</li>
                <li><strong>Data Profiling:</strong> Automated analysis of document characteristics and complexity</li>
                <li><strong>Parallel Processing:</strong> DASK workers evaluate multiple configurations simultaneously</li>
                <li><strong>Benchmark Matching:</strong> AI identifies most relevant external performance data</li>
                <li><strong>Optimization Engine:</strong> ML algorithms converge on optimal configuration</li>
                <li><strong>Production Deployment:</strong> Automated setup with monitoring and alerting</li>
            </ol>
            <h6>Quality Assurance:</h6>
            <ul>
                <li>Statistical validation of all recommendations</li>
                <li>Confidence intervals for performance predictions</li>
                <li>Automated rollback capabilities for failed deployments</li>
            </ul>
        `;
    }

    getROIDetails() {
        return `
            <h5 style="color: var(--color-success-bright);">Investment Protection Analysis</h5>
            <div style="background: var(--color-bg-3); padding: 16px; border-radius: 8px; margin: 16px 0;">
                <h6>Financial Impact Summary:</h6>
                <p><strong>For every $1M invested in RAG projects, save $300K-400K from avoided failures</strong></p>
            </div>
            <h6>Detailed ROI Breakdown:</h6>
            <ul>
                <li><strong>Failure Rate Reduction:</strong> 60% → 20% (66% improvement)</li>
                <li><strong>Time Savings:</strong> 2-4 weeks → 15 minutes per project</li>
                <li><strong>Resource Efficiency:</strong> 50% reduction in engineering overhead</li>
                <li><strong>Operational Costs:</strong> 60-80% reduction in compute and API expenses</li>
            </ul>
            <h6>Market Opportunity:</h6>
            <ul>
                <li><strong>RAG Market Growth:</strong> 45% CAGR, reaching $67B by 2030</li>
                <li><strong>Enterprise Adoption:</strong> 78% planning RAG implementations</li>
                <li><strong>Competitive Position:</strong> First-to-market in automated optimization</li>
            </ul>
            <div style="background: var(--color-bg-2); padding: 16px; border-radius: 8px; margin-top: 16px;">
                <h6>Payback Period:</h6>
                <p><strong>4-6 months</strong> for typical enterprise deployment</p>
            </div>
        `;
    }

    getArchitectureDetails() {
        return `
            <h5>Five-Layer Technical Architecture</h5>
            <div style="background: var(--color-bg-1); padding: 16px; border-radius: 8px; margin: 16px 0;">
                <h6>Proven Technology Stack:</h6>
                <p>Built on enterprise-grade, battle-tested technologies with proven scalability.</p>
            </div>
            <h6>Architecture Layers:</h6>
            <ol style="padding-left: 20px;">
                <li><strong>React UI Layer:</strong> Modern, responsive interface with real-time dashboards</li>
                <li><strong>FastAPI Services:</strong> High-performance microservices with auto-documentation</li>
                <li><strong>DASK Computing:</strong> Distributed processing with dynamic scaling</li>
                <li><strong>ML Optimization:</strong> Advanced algorithms with statistical validation</li>
                <li><strong>Data Management:</strong> Secure storage with benchmark integration</li>
            </ol>
            <h6>Deployment Characteristics:</h6>
            <ul>
                <li><strong>Containerized:</strong> Docker-based with Kubernetes orchestration</li>
                <li><strong>Auto-scaling:</strong> Dynamic resource allocation based on load</li>
                <li><strong>Cloud-agnostic:</strong> Deployable on AWS, Azure, GCP, or on-premises</li>
                <li><strong>Security:</strong> Enterprise-grade encryption and access controls</li>
            </ul>
            <h6>Performance Characteristics:</h6>
            <ul>
                <li><strong>Throughput:</strong> Handles 1000+ concurrent optimization requests</li>
                <li><strong>Latency:</strong> Sub-second response times for configuration recommendations</li>
                <li><strong>Reliability:</strong> 99.9% uptime with automated failover</li>
            </ul>
        `;
    }

    handleFeatureClick(event) {
        const feature = event.currentTarget;
        this.showToast('Feature details available in full technical proposal', 'info');
    }

    handleFeatureHover(event) {
        const feature = event.currentTarget;
        feature.style.transform = 'translateX(8px) scale(1.02)';
        feature.style.boxShadow = 'var(--shadow-md)';
    }

    handleStepClick(event) {
        const step = event.currentTarget;
        const stepText = step.querySelector('.step-content h4')?.textContent || 'Process Step';
        this.showToast(`${stepText}: Interactive demo available upon request`, 'info');
    }

    handleMetricClick(event) {
        const card = event.currentTarget;
        this.showToast('Detailed financial analysis available in comprehensive proposal', 'info');
    }

    // Animation Functions
    animateCounters() {
        const counters = document.querySelectorAll('.stat-number, .metric-value, .timeline-stat');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.animateCounter(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        });
        
        counters.forEach(counter => observer.observe(counter));
    }

    animateCounter(element) {
        const text = element.textContent;
        const numbers = text.match(/\d+/g);
        
        if (numbers && numbers.length > 0) {
            const targetNumber = parseInt(numbers[0]);
            if (targetNumber > 1) {
                let current = 0;
                const increment = targetNumber / 50;
                const timer = setInterval(() => {
                    current += increment;
                    if (current >= targetNumber) {
                        element.textContent = text;
                        clearInterval(timer);
                    } else {
                        element.textContent = text.replace(/\d+/, Math.floor(current));
                    }
                }, 30);
            }
        }
    }

    animateOnScroll() {
        const sections = document.querySelectorAll('.presentation-section');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, { threshold: 0.1 });
        
        sections.forEach((section, index) => {
            section.style.opacity = '0';
            section.style.transform = 'translateY(20px)';
            section.style.transition = `opacity 0.6s ease ${index * 0.1}s, transform 0.6s ease ${index * 0.1}s`;
            observer.observe(section);
        });
    }

    setupHoverEffects() {
        document.querySelectorAll('.feature-item').forEach(item => {
            item.addEventListener('mouseleave', () => {
                item.style.transform = '';
                item.style.boxShadow = '';
            });
        });
    }

    setupScrollAnimations() {
        // Smooth scrolling for any internal links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    addTooltips() {
        // Add informative tooltips to key elements
        const tooltips = {
            '.stat-number': 'Click section for detailed analysis',
            '.feature-icon': 'Core system component',
            '.step-icon': 'Process step details available',
            '.tech-icon': 'Proven technology stack'
        };
        
        Object.entries(tooltips).forEach(([selector, text]) => {
            document.querySelectorAll(selector).forEach(element => {
                element.title = text;
            });
        });
    }

    updateProgressIndicators() {
        // Add visual progress indicators
        const progressBar = document.createElement('div');
        progressBar.className = 'progress-indicator';
        progressBar.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 0%;
            height: 3px;
            background: var(--color-gradient-primary);
            z-index: 9999;
            transition: width 0.3s ease;
        `;
        document.body.appendChild(progressBar);
        
        window.addEventListener('scroll', () => {
            const scrollPercent = (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100;
            progressBar.style.width = `${Math.min(scrollPercent, 100)}%`;
        });
    }

    handleKeyboardShortcuts(event) {
        switch (event.key) {
            case 'Escape':
                this.closeModal();
                break;
            case 'p':
            case 'P':
                if (event.ctrlKey || event.metaKey) {
                    event.preventDefault();
                    this.handlePrint();
                }
                break;
            case 'd':
            case 'D':
                if (event.ctrlKey || event.metaKey) {
                    event.preventDefault();
                    this.handlePDFDownload();
                }
                break;
        }
    }

    // Modal Management
    showModal(title, content) {
        const modal = document.getElementById('detailModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalBody = document.getElementById('modalBody');
        
        if (modal && modalTitle && modalBody) {
            modalTitle.textContent = title;
            modalBody.innerHTML = content;
            modal.classList.remove('hidden');
            document.body.style.overflow = 'hidden';
        }
    }

    closeModal() {
        const modal = document.getElementById('detailModal');
        if (modal) {
            modal.classList.add('hidden');
            document.body.style.overflow = '';
        }
    }

    // Toast Notifications
    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        container.appendChild(toast);
        
        // Auto-remove after 4 seconds
        setTimeout(() => {
            toast.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => {
                if (toast.parentNode) {
                    container.removeChild(toast);
                }
            }, 300);
        }, 4000);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded - Initializing RAG Pipeline Optimization Presentation');
    new PresentationController();
});

// Add global error handling
window.addEventListener('error', (event) => {
    console.error('Application error:', event.error);
});

// Performance monitoring
window.addEventListener('load', () => {
    console.log('RAG Pipeline Optimization Presentation fully loaded');
});