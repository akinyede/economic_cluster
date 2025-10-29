// Analysis Mode Configuration and Management
const AnalysisModeManager = {
    // Mode definitions with clear parameters
    modes: {
        quick: {
            name: "Quick Analysis",
            icon: "fas fa-bolt",
            description: "Essential insights with intelligent clustering",
            estimatedTime: "Less than 30 minutes",
            features: [
                "Core business data from state registries",
                "ML-powered intelligent cluster discovery",
                "Top 3 optimal clusters discovered",
                "Patent search skipped by default for speed (can enable)",
                "Greedy optimizer enabled for faster clustering",
                "Predictive insights and confidence scores"
            ],
            parameters: {
                // Quick mode defaults: skip patent search and keep core sources
                data_sources: ['state_registrations', 'bls_employment'],
                use_ml_enhancement: true,  // Keep ML for intelligent clustering
                market_monitoring: false,
                sample_size: 2000,  // Increased for better representation
                skip_patents: true,
                quick_mode: true,
                runtime_options: {
                    disable_nsga2: true
                },
                algorithm_params: {
                    // Both num_clusters and cluster_size omitted for full automatic discovery
                }
            }
        },
        full: {
            name: "Full Analysis",
            icon: "fas fa-microscope",
            description: "Comprehensive deep dive with all data sources",
            estimatedTime: "2-3 hours",
            features: [
                "All data sources including patents & SBIR",
                "ML-enhanced predictions with network analysis",
                "Discover up to 5 optimal clusters",
                "Real-time market monitoring data",
                "Comprehensive recommendations for all stakeholders",
                "University partnership opportunities"
            ],
            parameters: {
                // Patents always included - Full mode searches all businesses
                data_sources: ['state_registrations', 'bls_employment', 'uspto_patents', 'sbir_awards', 'infrastructure_assets'],
                use_ml_enhancement: true,
                market_monitoring: true,
                sample_size: null,  // Analyze all businesses
                quick_mode: false,
                algorithm_params: {
                    // Both num_clusters and cluster_size omitted for full automatic discovery
                }
            }
        }
    },

    // Intelligent parameter detection
    detectOptimalMode: function(formData) {
        let timeEstimate = 5; // Base minutes
        let complexity = 0;

        // Counties selected
        const counties = $('.county-ks:checked, .county-mo:checked').length;
        timeEstimate += counties * 2;
        complexity += counties > 5 ? 2 : 1;

        // Data sources
        const dataSources = $('input[id^="source"]:checked').length;
        timeEstimate += dataSources * 5;
        complexity += dataSources > 3 ? 2 : 1;

        // Patent search time depends on mode
        const isQuickMode = $('#quickMode').is(':checked');
        if (isQuickMode) {
            // Skipped by default in quick mode
            timeEstimate += 0;
        } else {
            timeEstimate += 120; // 2 hours for all businesses
            complexity += 5;
        }

        // ML and market monitoring
        if ($('#mlEnhancement').val() === 'true') {
            timeEstimate += 10;
            complexity += 1;
        }
        if ($('#marketMonitoring').val() === 'true') {
            timeEstimate += 10;
            complexity += 1;
        }

        // Business filters
        const minEmployees = parseInt($('#minEmployees').val());
        const minRevenue = parseFloat($('#minRevenue').val());
        if (minEmployees < 10 && minRevenue < 0.5) {
            timeEstimate += 15; // More businesses to analyze
            complexity += 1;
        }

        return {
            recommendedMode: complexity > 5 ? 'full' : 'quick',
            estimatedMinutes: timeEstimate,
            complexityScore: complexity,
            factors: {
                hasPatents: !isQuickMode,
                largeGeography: counties > 5,
                manyDataSources: dataSources > 3,
                broadFilters: minEmployees < 10 && minRevenue < 0.5
            }
        };
    },

    // Apply mode settings to form
    applyMode: function(modeName) {
        const mode = this.modes[modeName];
        if (!mode) return;

        const params = mode.parameters;

        // Update form based on mode
        if (modeName === 'quick') {
            // Quick mode settings
            // Keep ML enabled and prefer enhanced KC models
            $('#mlEnhancement').val('enhanced');
            $('#marketMonitoring').val('false');
            
            // Limit data sources
            $('#sourceUSPTO').prop('checked', false);
            $('#sourceSBIR').prop('checked', false);
            $('#sourceEDCKC').prop('checked', false);
            
            // Select top 3 clusters
            $('.cluster-selection').prop('checked', false);
            $('#clusterLogistics, #clusterManufacturing, #clusterTechnology').prop('checked', true);
            
        } else {
            // Full mode settings
            // Prefer enhanced KC models for best accuracy
            $('#mlEnhancement').val('enhanced');
            $('#marketMonitoring').val('true');
            
            // Enable all data sources
            $('input[id^="source"]').prop('checked', true);
            
            // Select all clusters
            $('.cluster-selection').prop('checked', true);
        }

        // Update time estimate
        this.updateTimeEstimate();
        
        // Update form validation
        if (typeof updateFormProgress === 'function') {
            updateFormProgress();
        }
    },

    // Update time estimate display
    updateTimeEstimate: function() {
        const analysis = this.detectOptimalMode();
        const minutes = analysis.estimatedMinutes;
        
        let timeStr, badgeClass;
        if (minutes < 60) {
            timeStr = `~${Math.round(minutes)} minutes`;
            badgeClass = minutes < 15 ? 'bg-success' : 'bg-info';
        } else {
            const hours = Math.floor(minutes / 60);
            const mins = Math.round(minutes % 60);
            timeStr = `~${hours}h ${mins}m`;
            badgeClass = 'bg-warning';
        }
        
        $('#estimatedTime').text(timeStr).removeClass('bg-success bg-info bg-warning').addClass(badgeClass);
        
        // Mode recommendation banner is not needed for the streamlined flow
        $('#modeRecommendation').hide();
        
        return analysis;
    },

    // Initialize the mode selector
    init: function() {
        // Bind events
        $('input[name="analysisMode"]').on('change', (e) => {
            const mode = $(e.target).attr('id') === 'quickMode' ? 'quick' : 'full';
            this.applyMode(mode);
            this.updateModeDisplay(mode);
        });

        // Update time when relevant parameters change
        $('.county-ks, .county-mo, input[id^="source"], #mlEnhancement, #marketMonitoring').on('change', () => {
            this.updateTimeEstimate();
        });
        
        $('#minEmployees, #maxEmployees, #minRevenue, #businessAge').on('input', () => {
            this.updateTimeEstimate();
        });

        // Advanced options
        $('#limitBusinesses, #disableML, #cacheOnly').on('change', () => {
            this.updateTimeEstimate();
        });

        // Initial setup
        this.updateTimeEstimate();
    },

    // Update mode display
    updateModeDisplay: function(modeName) {
        const mode = this.modes[modeName];
        $('#modeTitle').text(mode.name + ' Selected');
        $('#modeFeatures').html(mode.features.map(f => `<li>${f}</li>`).join(''));
        $('#modeDescription').text(mode.description);
    }
};

// Extend the runAnalysis function to include mode parameters
window.runAnalysisWithMode = function() {
    // Get current mode
    const isQuickMode = $('#quickMode').is(':checked');
    const modeParams = AnalysisModeManager.modes[isQuickMode ? 'quick' : 'full'].parameters;
    
    // Gather standard parameters
    const params = typeof gatherParameters === 'function' ? gatherParameters() : {};
    
    // Merge with mode parameters
    Object.assign(params, modeParams);
    
    // Apply any advanced option overrides
    
    if ($('#limitBusinesses').is(':checked')) {
        params.sample_size = 2000;
    }
    
    // Note: We keep ML enabled even if user tries to disable it in quick mode
    // This ensures intelligent clustering always works
    params.use_ml_enhancement = true;
    
    if ($('#cacheOnly').is(':checked')) {
        params.use_cached_data = true;
    }
    
    // Log parameters for debugging
    console.log('Analysis parameters:', params);
    
    // Prefer delegating to the main runner to avoid duplicate submission logic
    try {
        if (typeof startAnalysisWithParams === 'function') {
            return startAnalysisWithParams(Object.assign({}, params));
        }
        if (typeof runAnalysis === 'function') {
            return runAnalysis();
        }
    } catch (e) {
        console.warn('Delegation to main runner failed, falling back to direct request:', e);
    }
    
    // Show loading with mode info
    if (typeof showLoading === 'function') {
        showLoading();
    }
    
    const loadingMessage = isQuickMode ? 
        'Running quick analysis with intelligent ML clustering...' : 
        'Running comprehensive analysis with full data sources...';
    
    $('#loadingOverlay p').text(loadingMessage);
    
    // Make the API call
    $.ajax({
        url: '/api/run_analysis_async',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(params),
        success: function(response) {
            if (response.status === 'redirect') {
                // Handle async redirect for complex analyses
                window.location.href = response.redirect_url;
            } else if (response.status === 'success') {
                // Direct results for quick analyses
                displayResults(response.results, response.visualizations);
                hideLoading();
            } else if (response.status === 'Analysis started' && response.task_id) {
                // Async path: begin progress monitoring like the main runner
                try {
                    if (typeof monitorAsyncProgress === 'function') {
                        monitorAsyncProgress(response.task_id);
                    }
                } finally {
                    hideLoading();
                }
            }
        },
        error: function(xhr, status, error) {
            hideLoading();
            console.error('Analysis error:', error);
            // Handle error display
        }
    });
};

// Initialize on document ready
$(document).ready(function() {
    AnalysisModeManager.init();
});
