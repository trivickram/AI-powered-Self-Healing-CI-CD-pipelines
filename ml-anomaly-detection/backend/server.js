const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const axios = require('axios');
const { spawn } = require('child_process');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5000';

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// In-memory storage for anomalies and healing actions
let anomalies = [];
let healingActions = [];
let pipelineHealth = 'healthy';

// Helper function to add anomaly
function addAnomaly(logMessage, prediction, timestamp) {
    anomalies.unshift({
        id: Date.now(),
        logMessage: logMessage.substring(0, 100) + '...',
        prediction,
        timestamp: timestamp || new Date().toISOString(),
        status: 'detected'
    });
    
    // Keep only last 50 anomalies
    if (anomalies.length > 50) {
        anomalies = anomalies.slice(0, 50);
    }
}

// Helper function to add healing action
function addHealingAction(action, result, anomalyId) {
    healingActions.unshift({
        id: Date.now(),
        anomalyId,
        action,
        result,
        timestamp: new Date().toISOString(),
        status: result === 'success' ? 'successful' : 'failed'
    });
    
    // Keep only last 50 healing actions
    if (healingActions.length > 50) {
        healingActions = healingActions.slice(0, 50);
    }
}

// Function to determine healing action based on log content
function determineHealingAction(logMessage) {
    const lowerLog = logMessage.toLowerCase();
    
    if (lowerLog.includes('dependency') || lowerLog.includes('module not found') || lowerLog.includes('npm')) {
        return 'clear_cache';
    } else if (lowerLog.includes('network') || lowerLog.includes('timeout') || lowerLog.includes('connection')) {
        return 'retry_job';
    } else if (lowerLog.includes('test') || lowerLog.includes('assertion') || lowerLog.includes('expect')) {
        return 'rerun_tests';
    } else if (lowerLog.includes('build') || lowerLog.includes('compile')) {
        return 'rebuild';
    } else if (lowerLog.includes('deploy') || lowerLog.includes('upload')) {
        return 'redeploy';
    } else {
        return 'investigate';
    }
}

// Function to execute healing action
async function executeHealingAction(action, anomalyId) {
    console.log(`🔧 Executing healing action: ${action}`);
    
    try {
        let result = 'success';
        let details = '';
        
        switch (action) {
            case 'clear_cache':
                // Simulate clearing cache
                await new Promise(resolve => setTimeout(resolve, 1000));
                details = 'Cleared npm cache and node_modules';
                console.log('✅ Cache cleared successfully');
                break;
                
            case 'retry_job':
                // Simulate retrying job
                await new Promise(resolve => setTimeout(resolve, 1500));
                details = 'Retried job execution with exponential backoff';
                console.log('✅ Job retried successfully');
                break;
                
            case 'rerun_tests':
                // Simulate rerunning tests
                await new Promise(resolve => setTimeout(resolve, 2000));
                details = 'Reran test suite with fresh environment';
                console.log('✅ Tests rerun successfully');
                break;
                
            case 'rebuild':
                // Simulate rebuild
                await new Promise(resolve => setTimeout(resolve, 3000));
                details = 'Rebuilt project from clean state';
                console.log('✅ Rebuild completed successfully');
                break;
                
            case 'redeploy':
                // Simulate redeployment
                await new Promise(resolve => setTimeout(resolve, 2500));
                details = 'Redeployed application with updated configuration';
                console.log('✅ Redeployment completed successfully');
                break;
                
            default:
                details = 'Manual investigation required - anomaly logged for review';
                console.log('⚠️ Manual investigation flagged');
                break;
        }
        
        addHealingAction(`${action}: ${details}`, result, anomalyId);
        return { success: true, action, details };
        
    } catch (error) {
        console.error(`❌ Healing action failed: ${error.message}`);
        addHealingAction(`${action}: Failed - ${error.message}`, 'failed', anomalyId);
        return { success: false, action, error: error.message };
    }
}

// Routes

// POST /logs - Receive pipeline logs and analyze for anomalies
app.post('/logs', async (req, res) => {
    try {
        const { log_message, timestamp, status, job_id } = req.body;
        
        if (!log_message) {
            return res.status(400).json({ 
                error: 'log_message is required' 
            });
        }
        
        console.log(`📋 Received log for analysis: ${log_message.substring(0, 50)}...`);
        
        // Call ML service for prediction
        const mlResponse = await axios.post(`${ML_SERVICE_URL}/predict`, {
            log_message: log_message
        }, {
            timeout: 10000
        });
        
        const prediction = mlResponse.data.prediction;
        const confidence = mlResponse.data.confidence || 0;
        
        console.log(`🤖 ML Prediction: ${prediction} (confidence: ${confidence})`);
        
        // If anomaly detected, trigger healing
        if (prediction === 'anomaly') {
            pipelineHealth = 'unhealthy';
            
            const anomalyId = Date.now();
            addAnomaly(log_message, prediction, timestamp);
            
            // Trigger healing action
            console.log('🚨 Anomaly detected - initiating healing process');
            const healingAction = determineHealingAction(log_message);
            
            // Execute healing in background
            setImmediate(async () => {
                await executeHealingAction(healingAction, anomalyId);
            });
            
            res.json({
                success: true,
                anomaly_detected: true,
                prediction,
                confidence,
                healing_action: healingAction,
                anomaly_id: anomalyId,
                message: 'Anomaly detected - healing process initiated'
            });
        } else {
            // Normal log - update health if we were unhealthy
            if (pipelineHealth === 'unhealthy' && anomalies.length > 0) {
                // Check if recent anomalies have been addressed
                const recentAnomalies = anomalies.filter(a => 
                    (Date.now() - new Date(a.timestamp).getTime()) < 300000 // 5 minutes
                );
                
                if (recentAnomalies.length === 0) {
                    pipelineHealth = 'healthy';
                    console.log('✅ Pipeline health restored');
                }
            }
            
            res.json({
                success: true,
                anomaly_detected: false,
                prediction,
                confidence,
                message: 'Log processed - no anomaly detected'
            });
        }
        
    } catch (error) {
        console.error('❌ Error processing log:', error.message);
        
        // If ML service is down, use fallback detection
        if (error.code === 'ECONNREFUSED' || error.response?.status >= 500) {
            console.log('🔄 ML service unavailable - using fallback detection');
            
            const fallbackPrediction = determineFallbackAnomaly(req.body.log_message);
            
            if (fallbackPrediction === 'anomaly') {
                const anomalyId = Date.now();
                addAnomaly(req.body.log_message, 'anomaly (fallback)', req.body.timestamp);
                
                const healingAction = determineHealingAction(req.body.log_message);
                setImmediate(async () => {
                    await executeHealingAction(healingAction, anomalyId);
                });
                
                return res.json({
                    success: true,
                    anomaly_detected: true,
                    prediction: 'anomaly',
                    confidence: 0.7,
                    healing_action: healingAction,
                    anomaly_id: anomalyId,
                    message: 'Anomaly detected using fallback detection',
                    fallback_used: true
                });
            }
        }
        
        res.status(500).json({ 
            error: 'Failed to process log', 
            details: error.message 
        });
    }
});

// Fallback anomaly detection using keyword analysis
function determineFallbackAnomaly(logMessage) {
    const errorKeywords = [
        'error', 'failed', 'exception', 'timeout', 'connection refused',
        'not found', 'permission denied', 'access denied', 'fatal',
        'critical', 'panic', 'crash', 'abort', 'kill', 'segmentation fault'
    ];
    
    const lowerLog = logMessage.toLowerCase();
    
    for (const keyword of errorKeywords) {
        if (lowerLog.includes(keyword)) {
            return 'anomaly';
        }
    }
    
    return 'normal';
}

// POST /heal - Manual healing trigger
app.post('/heal', async (req, res) => {
    try {
        const { action, anomaly_id, log_message } = req.body;
        
        let healingAction = action;
        if (!healingAction && log_message) {
            healingAction = determineHealingAction(log_message);
        }
        
        if (!healingAction) {
            return res.status(400).json({ 
                error: 'Healing action or log_message required' 
            });
        }
        
        console.log(`🔧 Manual healing requested: ${healingAction}`);
        
        const result = await executeHealingAction(healingAction, anomaly_id);
        
        res.json({
            success: true,
            healing_result: result,
            message: 'Healing action executed'
        });
        
    } catch (error) {
        console.error('❌ Error executing healing:', error.message);
        res.status(500).json({ 
            error: 'Failed to execute healing action', 
            details: error.message 
        });
    }
});

// GET /status - Get pipeline health and recent activity
app.get('/status', (req, res) => {
    const recentAnomalies = anomalies.slice(0, 10);
    const recentHealingActions = healingActions.slice(0, 10);
    
    // Calculate health metrics
    const totalAnomalies = anomalies.length;
    const successfulHealing = healingActions.filter(a => a.status === 'successful').length;
    const healingSuccessRate = healingActions.length > 0 ? 
        Math.round((successfulHealing / healingActions.length) * 100) : 100;
    
    res.json({
        success: true,
        pipeline_health: pipelineHealth,
        health_metrics: {
            total_anomalies: totalAnomalies,
            successful_healing: successfulHealing,
            healing_success_rate: healingSuccessRate,
            last_check: new Date().toISOString()
        },
        recent_anomalies: recentAnomalies,
        recent_healing_actions: recentHealingActions,
        ml_service_status: 'unknown' // Could ping ML service here
    });
});

// GET /health - Simple health check
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        service: 'self-healing-backend',
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
    });
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('❌ Unhandled error:', error);
    res.status(500).json({
        error: 'Internal server error',
        message: error.message
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Self-Healing Backend Server running on port ${PORT}`);
    console.log(`📡 ML Service URL: ${ML_SERVICE_URL}`);
    console.log(`🔗 Status endpoint: http://localhost:${PORT}/status`);
    console.log(`💚 Health check: http://localhost:${PORT}/health`);
});

module.exports = app;
