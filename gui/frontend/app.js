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
    // Bow River Basin fallback coordinates (used only if GeoTIFF bounds unavailable)
    MAP_CENTER: [51.0447, -114.0719],
    MAP_ZOOM: 10,
    GEOTIFF_URL: 'http://localhost:5000/api/geotiff'
};

// State
let classificationResults = null;
let map = null;
let chart = null;

// DOM Elements
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
    fetchResults();
    console.log('🌿 Wetland Mapping Application Initialized');
}

// Event Listeners
function setupEventListeners() {
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

// Fetch Classification Results from Backend
async function fetchResults() {
    try {
        showLoading(true);
        updateProgress(10);

        const startTime = performance.now();

        updateProgress(30);

        const response = await fetch(`${CONFIG.API_BASE_URL}/api/results`);

        updateProgress(70);

        if (!response.ok) {
            throw new Error('Could not load results. Please check if the backend server is running.');
        }

        const results = await response.json();
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(2);

        updateProgress(100);

        classificationResults = {
            ...results,
            processingTime
        };

        displayResults(classificationResults);

    } catch (error) {
        console.error('Error fetching results:', error);
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
                        label: function (context) {
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
                        callback: function (value) {
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

// Hex colour string → [r, g, b] array (values 0-255)
function hexToRgb(hex) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return [r, g, b];
}

// Build a lookup: class id (0-5) → [r, g, b]
const CLASS_COLORS = Object.fromEntries(
    Object.entries(CONFIG.WETLAND_CLASSES).map(([id, cls]) => [Number(id), hexToRgb(cls.color)])
);

// Active GeoRasterLayer reference (kept so we can remove/replace it)
let geotiffLayer = null;

// Show Map Visualization — fetches GeoTIFF and renders it on the Leaflet map
async function showMapVisualization(results) {
    mapPlaceholder.classList.add('hidden');
    mapElement.classList.remove('hidden');

    setTimeout(() => map.invalidateSize(), 100);

    if (!results.geotiff_ready) {
        console.warn('⚠️ GeoTIFF not ready — skipping map overlay');
        return;
    }

    try {
        console.log('🗺️ Fetching GeoTIFF from backend...');
        const response = await fetch(CONFIG.GEOTIFF_URL);
        if (!response.ok) throw new Error(`GeoTIFF fetch failed: ${response.status}`);

        const arrayBuffer = await response.arrayBuffer();
        const georaster = await parseGeoraster(arrayBuffer);

        // Remove previous overlay if re-rendering
        if (geotiffLayer) {
            map.removeLayer(geotiffLayer);
        }

        geotiffLayer = new GeoRasterLayer({
            georaster,
            opacity: 0.75,
            pixelValuesToColorFn: (values) => {
                const classId = values[0];
                // 255 = nodata — render as transparent
                if (classId === 255 || classId === undefined) return null;
                const rgb = CLASS_COLORS[classId];
                if (!rgb) return null;
                return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
            },
            resolution: 256,  // tile resolution (pixels); higher = sharper but slower
        });

        geotiffLayer.addTo(map);

        // Fit the map view to the raster's actual geographic extent
        map.fitBounds(geotiffLayer.getBounds());

        console.log('✅ GeoTIFF overlay rendered on map');

    } catch (err) {
        console.error('❌ GeoTIFF overlay failed:', err);
        showNotification('Could not render map overlay: ' + err.message, 'error');
    }
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
}

// Utility Functions
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
    const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
    console.log(`${icon} ${message}`);
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
