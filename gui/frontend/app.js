// Configuration
const CONFIG = {
    API_BASE_URL: 'http://localhost:5000',
    WETLAND_CLASSES: {
        0: { name: 'Background', color: '#1a1a2e' },
        1: { name: 'Marsh', color: '#16c79a' },
        2: { name: 'Swamp', color: '#43b581' },
        3: { name: 'Fen', color: '#7289da' },
        4: { name: 'Bog', color: '#faa61a' },
        5: { name: 'Open Water', color: '#ee5a6f' }
    },
    // Bow River Basin coordinates
    MAP_CENTER: [51.0447, -114.0719],
    MAP_ZOOM: 10
};

// State
let currentFile = null;
let classificationResults = null;
let map = null;
let chart = null;

// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const classifyBtn = document.getElementById('classifyBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const progressFill = document.getElementById('progressFill');
const resultsSection = document.getElementById('resultsSection');
const mapPlaceholder = document.getElementById('mapPlaceholder');
const mapElement = document.getElementById('map');
const exportCSV = document.getElementById('exportCSV');
const exportJSON = document.getElementById('exportJSON');
const exportPNG = document.getElementById('exportPNG');

// Initialize Application
function init() {
    setupEventListeners();
    initializeMap();
    console.log('🌿 Wetland Mapping Application Initialized');
}

// Event Listeners
function setupEventListeners() {
    // Upload Zone Click
    uploadZone.addEventListener('click', () => fileInput.click());
    
    // File Input Change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and Drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect({ target: { files } });
        }
    });
    
    // Classify Button
    classifyBtn.addEventListener('click', handleClassify);
    
    // Export Buttons
    exportCSV.addEventListener('click', () => exportResults('csv'));
    exportJSON.addEventListener('click', () => exportResults('json'));
    exportPNG.addEventListener('click', () => exportMapImage());
}

// Initialize Leaflet Map
function initializeMap() {
    map = L.map('map', {
        center: CONFIG.MAP_CENTER,
        zoom: CONFIG.MAP_ZOOM,
        zoomControl: true
    });
    
    // Add OpenStreetMap tiles
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19
    }).addTo(map);
    
    // Add marker for Bow River Basin
    const marker = L.marker(CONFIG.MAP_CENTER).addTo(map);
    marker.bindPopup('<strong>Bow River Basin</strong><br>Alberta, Canada');
}

// File Handling
function handleFileSelect(event) {
    const file = event.target.files[0];
    
    if (!file) return;
    
    if (!file.name.endsWith('.npz') && !file.name.endsWith('.npy')) {
        showNotification('Please upload a valid .npz or .npy file', 'error');
        return;
    }
    
    currentFile = file;
    
    // Update UI
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.classList.remove('hidden');
    classifyBtn.disabled = false;
    
    console.log('📁 File selected:', file.name);
}

// Classification
async function handleClassify() {
    if (!currentFile) return;
    
    try {
        showLoading(true);
        updateProgress(10);
        
        const startTime = performance.now();
        
        // Prepare form data
        const formData = new FormData();
        formData.append('file', currentFile);
        
        updateProgress(30);
        
        // Make API request
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/predict`, {
            method: 'POST',
            body: formData
        });
        
        updateProgress(70);
        
        if (!response.ok) {
            throw new Error('Classification failed. Please check if the backend server is running.');
        }
        
        const results = await response.json();
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(2);
        
        updateProgress(100);
        
        // Store results
        classificationResults = {
            ...results,
            processingTime
        };
        
        // Display results
        displayResults(classificationResults);
        
        showNotification('Classification completed successfully!', 'success');
        
    } catch (error) {
        console.error('Classification error:', error);
        showNotification(error.message, 'error');
    } finally {
        showLoading(false);
    }
}

// Display Results
function displayResults(results) {
    // Show results section
    resultsSection.classList.add('active');
    
    // Update statistics
    document.getElementById('statTotal').textContent = formatNumber(results.total_samples || 0);
    document.getElementById('statAccuracy').textContent = results.confidence ? `${(results.confidence * 100).toFixed(1)}%` : 'N/A';
    document.getElementById('statTime').textContent = `${results.processingTime}s`;
    
    // Update legend with class distribution
    if (results.class_distribution) {
        Object.keys(results.class_distribution).forEach(classId => {
            const count = results.class_distribution[classId];
            const percentage = ((count / results.total_samples) * 100).toFixed(1);
            const legendValue = document.querySelector(`.legend-value[data-class="${classId}"]`);
            if (legendValue) {
                legendValue.textContent = `${percentage}%`;
            }
        });
    }
    
    // Update chart
    updateChart(results.class_distribution);
    
    // Update map visualization
    showMapVisualization(results);
    
    console.log('✅ Results displayed');
}

// Update Chart
function updateChart(distribution) {
    const ctx = document.getElementById('distributionChart');
    
    if (chart) {
        chart.destroy();
    }
    
    const labels = Object.keys(CONFIG.WETLAND_CLASSES).map(id => CONFIG.WETLAND_CLASSES[id].name);
    const data = Object.keys(CONFIG.WETLAND_CLASSES).map(id => distribution[id] || 0);
    const colors = Object.keys(CONFIG.WETLAND_CLASSES).map(id => CONFIG.WETLAND_CLASSES[id].color);
    
    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Sample Count',
                data: data,
                backgroundColor: colors,
                borderColor: colors.map(c => c + '80'),
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(26, 26, 46, 0.95)',
                    titleColor: '#ffffff',
                    bodyColor: '#b8b9cf',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            return `Count: ${formatNumber(context.parsed.y)}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#7885a3',
                        callback: function(value) {
                            return formatNumber(value);
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#7885a3'
                    }
                }
            }
        }
    });
}

// Show Map Visualization
function showMapVisualization(results) {
    mapPlaceholder.classList.add('hidden');
    mapElement.classList.remove('hidden');
    
    // Invalidate size to ensure proper rendering
    setTimeout(() => {
        map.invalidateSize();
    }, 100);
    
    // Add classification overlay (simplified visualization)
    // In a real implementation, this would render actual classified tiles
    if (results.predictions && results.coordinates) {
        // Create a heat map or marker cluster based on classification results
        // For now, we'll add a simple info popup
        const infoPopup = L.popup()
            .setLatLng(CONFIG.MAP_CENTER)
            .setContent(`
                <div style="text-align: center;">
                    <strong>Classification Complete</strong><br>
                    <small>${formatNumber(results.total_samples)} samples processed</small>
                </div>
            `)
            .openOn(map);
    }
    
    console.log('🗺️ Map visualization updated');
}

// Export Results
function exportResults(format) {
    if (!classificationResults) {
        showNotification('No results to export', 'error');
        return;
    }
    
    let content, filename, mimeType;
    
    if (format === 'csv') {
        content = generateCSV(classificationResults);
        filename = 'wetland_classification.csv';
        mimeType = 'text/csv';
    } else if (format === 'json') {
        content = JSON.stringify(classificationResults, null, 2);
        filename = 'wetland_classification.json';
        mimeType = 'application/json';
    }
    
    downloadFile(content, filename, mimeType);
    showNotification(`Exported as ${format.toUpperCase()}`, 'success');
}

function generateCSV(results) {
    let csv = 'Class ID,Class Name,Sample Count,Percentage\n';
    
    Object.keys(CONFIG.WETLAND_CLASSES).forEach(classId => {
        const count = results.class_distribution[classId] || 0;
        const percentage = ((count / results.total_samples) * 100).toFixed(2);
        const className = CONFIG.WETLAND_CLASSES[classId].name;
        csv += `${classId},${className},${count},${percentage}%\n`;
    });
    
    return csv;
}

function exportMapImage() {
    showNotification('Map export functionality coming soon!', 'info');
    // This would use leaflet-image or similar library to export the map
}

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function formatNumber(num) {
    return new Intl.NumberFormat('en-US').format(num);
}

function showLoading(show) {
    if (show) {
        loadingOverlay.classList.remove('hidden');
        updateProgress(0);
    } else {
        setTimeout(() => {
            loadingOverlay.classList.add('hidden');
        }, 500);
    }
}

function updateProgress(percent) {
    progressFill.style.width = `${percent}%`;
}

function showNotification(message, type = 'info') {
    // Simple console notification for now
    // In production, this would show a toast notification
    const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
    console.log(`${icon} ${message}`);
    
    // Could implement a toast notification here
    alert(message);
}

function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// Initialize on DOM load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Demo mode for testing without backend
const DEMO_MODE = false;

if (DEMO_MODE) {
    // Simulate classification results for testing
    setTimeout(() => {
        classificationResults = {
            total_samples: 150000,
            confidence: 0.87,
            processingTime: '2.35',
            class_distribution: {
                0: 45000,
                1: 32000,
                2: 28000,
                3: 18000,
                4: 15000,
                5: 12000
            }
        };
        displayResults(classificationResults);
    }, 2000);
}
