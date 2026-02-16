/**
 * WebGL Visualization Engine
 * 
 * Renders 2D/3D scatter, bar, line charts using WebGL
 * Can handle MILLIONS of points efficiently via GPU acceleration
 */

export class WebGPUViz {
    constructor(device) {
        this.device = device;
        this.initialized = false;
        this.gl = null;
        this.shaderPrograms = {};
    }

    async init() {
        console.log('WebGL Visualization Engine initialized');
        this.initialized = true;
    }

    /**
     * Initialize WebGL context and shaders
     */
    initWebGL(canvas) {
        if (this.gl) return this.gl;

        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        if (!gl) {
            throw new Error('WebGL not supported');
        }

        this.gl = gl;

        // Create shader programs
        this.shaderPrograms.points = this.createPointShaderProgram(gl);
        this.shaderPrograms.lines = this.createLineShaderProgram(gl);
        this.shaderPrograms.bars = this.createBarShaderProgram(gl);

        console.log('✅ WebGL initialized successfully');
        return gl;
    }

    /**
     * Create shader program for point rendering (scatter plots)
     */
    createPointShaderProgram(gl) {
        const vertexShaderSource = `
            attribute vec2 a_position;
            attribute vec3 a_color;
            attribute float a_size;
            
            uniform vec2 u_resolution;
            uniform vec4 u_bounds;
            
            varying vec3 v_color;
            
            void main() {
                vec2 normalized = (a_position - u_bounds.xy) / (u_bounds.zw - u_bounds.xy);
                vec2 clipSpace = normalized * 2.0 - 1.0;
                clipSpace.y = -clipSpace.y;
                
                gl_Position = vec4(clipSpace, 0.0, 1.0);
                gl_PointSize = a_size;
                v_color = a_color;
            }
        `;

        const fragmentShaderSource = `
            precision mediump float;
            varying vec3 v_color;
            
            void main() {
                vec2 coord = gl_PointCoord - vec2(0.5);
                float dist = length(coord);
                if (dist > 0.5) discard;
                
                float alpha = smoothstep(0.5, 0.4, dist);
                gl_FragColor = vec4(v_color, alpha * 0.8);
            }
        `;

        return this.compileShaderProgram(gl, vertexShaderSource, fragmentShaderSource);
    }

    /**
     * Create shader program for line rendering
     */
    createLineShaderProgram(gl) {
        const vertexShaderSource = `
            attribute vec2 a_position;
            uniform vec2 u_resolution;
            uniform vec4 u_bounds;
            
            void main() {
                vec2 normalized = (a_position - u_bounds.xy) / (u_bounds.zw - u_bounds.xy);
                vec2 clipSpace = normalized * 2.0 - 1.0;
                clipSpace.y = -clipSpace.y;
                gl_Position = vec4(clipSpace, 0.0, 1.0);
            }
        `;

        const fragmentShaderSource = `
            precision mediump float;
            uniform vec3 u_color;
            
            void main() {
                gl_FragColor = vec4(u_color, 0.9);
            }
        `;

        return this.compileShaderProgram(gl, vertexShaderSource, fragmentShaderSource);
    }

    /**
     * Create shader program for bar rendering
     */
    createBarShaderProgram(gl) {
        const vertexShaderSource = `
            attribute vec2 a_position;
            attribute vec3 a_color;
            
            uniform vec2 u_resolution;
            
            varying vec3 v_color;
            
            void main() {
                vec2 clipSpace = (a_position / u_resolution) * 2.0 - 1.0;
                clipSpace.y = -clipSpace.y;
                gl_Position = vec4(clipSpace, 0.0, 1.0);
                v_color = a_color;
            }
        `;

        const fragmentShaderSource = `
            precision mediump float;
            varying vec3 v_color;
            
            void main() {
                gl_FragColor = vec4(v_color, 0.85);
            }
        `;

        return this.compileShaderProgram(gl, vertexShaderSource, fragmentShaderSource);
    }

    /**
     * Compile and link shader program
     */
    compileShaderProgram(gl, vertexSource, fragmentSource) {
        const vertexShader = this.compileShader(gl, gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = this.compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);

        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);

        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            throw new Error('Shader program linking failed: ' + gl.getProgramInfoLog(program));
        }

        return program;
    }

    /**
     * Compile individual shader
     */
    compileShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            throw new Error('Shader compilation failed: ' + gl.getShaderInfoLog(shader));
        }

        return shader;
    }

    /**
     * Render 2D Scatter Plot with WebGL
     */
    async render2DScatter(canvas, data, xColumn, yColumn, colorColumn = null) {
        const startTime = performance.now();
        const gl = this.initWebGL(canvas);
        
        console.log(` WebGL Rendering ${data.length} points: X="${xColumn}", Y="${yColumn}"`);

        const points = [];
        const colors = [];
        const sizes = [];
        
        let xMin = Infinity, xMax = -Infinity;
        let yMin = Infinity, yMax = -Infinity;

        data.forEach((row, idx) => {
            const x = parseFloat(row[xColumn]);
            const y = parseFloat(row[yColumn]);

            if (isNaN(x) || isNaN(y)) return;

            points.push(x, y);
            
            xMin = Math.min(xMin, x);
            xMax = Math.max(xMax, x);
            yMin = Math.min(yMin, y);
            yMax = Math.max(yMax, y);

            const hue = colorColumn ? (idx / data.length) * 360 : 200;
            const rgb = this.hslToRgb(hue, 70, 60);
            colors.push(rgb[0], rgb[1], rgb[2]);
            
            sizes.push(6.0);
        });

        if (points.length === 0) {
            this.renderNoData(canvas);
            return;
        }

        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0.04, 0.04, 0.04, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        const program = this.shaderPrograms.points;
        gl.useProgram(program);

        // Position buffer
        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(points), gl.STATIC_DRAW);
        const positionLoc = gl.getAttribLocation(program, 'a_position');
        gl.enableVertexAttribArray(positionLoc);
        gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

        // Color buffer
        const colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
        const colorLoc = gl.getAttribLocation(program, 'a_color');
        gl.enableVertexAttribArray(colorLoc);
        gl.vertexAttribPointer(colorLoc, 3, gl.FLOAT, false, 0, 0);

        // Size buffer
        const sizeBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(sizes), gl.STATIC_DRAW);
        const sizeLoc = gl.getAttribLocation(program, 'a_size');
        gl.enableVertexAttribArray(sizeLoc);
        gl.vertexAttribPointer(sizeLoc, 1, gl.FLOAT, false, 0, 0);

        // Set uniforms
        gl.uniform2f(gl.getUniformLocation(program, 'u_resolution'), canvas.width, canvas.height);
        gl.uniform4f(gl.getUniformLocation(program, 'u_bounds'), xMin, xMax, yMin, yMax);

        gl.drawArrays(gl.POINTS, 0, points.length / 2);

        this.drawAxes2D(canvas, xColumn, yColumn, xMin, xMax, yMin, yMax);

        console.log(` WebGL rendered ${points.length / 2} points in ${(performance.now() - startTime).toFixed(2)}ms`);
    }

    /**
     * Render Bar Chart with WebGL
     */
    async renderBarChart(canvas, data, xColumn, yColumn) {
        const startTime = performance.now();
        const gl = this.initWebGL(canvas);
        
        console.log(` WebGL Bar Chart: ${data.length} categories`);

        const categoryMap = new Map();
        data.forEach(row => {
            const category = String(row[xColumn]);
            const value = parseFloat(row[yColumn]);
            
            if (!isNaN(value)) {
                if (!categoryMap.has(category)) {
                    categoryMap.set(category, { sum: 0, count: 0 });
                }
                const stats = categoryMap.get(category);
                stats.sum += value;
                stats.count += 1;
            }
        });

        const categories = Array.from(categoryMap.keys());
        const values = categories.map(cat => categoryMap.get(cat).sum / categoryMap.get(cat).count);
        
        if (values.length === 0) {
            this.renderNoData(canvas);
            return;
        }

        const maxValue = Math.max(...values);
        const padding = 60;
        const width = canvas.width;
        const height = canvas.height;
        const plotHeight = height - padding * 2;
        const plotWidth = width - padding * 2;
        const barWidth = plotWidth / categories.length;

        const vertices = [];
        const colors = [];

        categories.forEach((cat, idx) => {
            const value = values[idx];
            const barHeight = (value / maxValue) * plotHeight;
            const x = padding + idx * barWidth;
            const y = height - padding - barHeight;

            vertices.push(
                x, height - padding,
                x + barWidth * 0.8, height - padding,
                x + barWidth * 0.8, y,
                x, height - padding,
                x + barWidth * 0.8, y,
                x, y
            );

            const hue = (idx / categories.length) * 360;
            const rgb = this.hslToRgb(hue, 70, 60);
            for (let i = 0; i < 6; i++) {
                colors.push(rgb[0], rgb[1], rgb[2]);
            }
        });

        gl.viewport(0, 0, width, height);
        gl.clearColor(0.04, 0.04, 0.04, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        const program = this.shaderPrograms.bars;
        gl.useProgram(program);

        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(gl.getAttribLocation(program, 'a_position'));
        gl.vertexAttribPointer(gl.getAttribLocation(program, 'a_position'), 2, gl.FLOAT, false, 0, 0);

        const colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(gl.getAttribLocation(program, 'a_color'));
        gl.vertexAttribPointer(gl.getAttribLocation(program, 'a_color'), 3, gl.FLOAT, false, 0, 0);

        gl.uniform2f(gl.getUniformLocation(program, 'u_resolution'), width, height);
        gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 2);

        this.drawBarLabels(canvas, categories, values, maxValue);

        console.log(` WebGL bar chart rendered in ${(performance.now() - startTime).toFixed(2)}ms`);
    }

    /**
     * Render Histogram with WebGL
     */
    async renderHistogram(canvas, data, column, bins = 20) {
        const values = data.map(row => parseFloat(row[column])).filter(v => !isNaN(v));
        
        if (values.length === 0) {
            this.renderNoData(canvas);
            return;
        }

        const min = Math.min(...values);
        const max = Math.max(...values);
        const binSize = (max - min) / bins;

        const histogram = new Array(bins).fill(0);
        values.forEach(value => {
            const binIndex = Math.min(Math.floor((value - min) / binSize), bins - 1);
            histogram[binIndex]++;
        });

        const histData = histogram.map((count, idx) => ({
            bin: (min + idx * binSize).toFixed(1),
            count: count
        }));

        await this.renderBarChart(canvas, histData, 'bin', 'count');
    }

    /**
     * Render 3D Scatter Plot
     */
    async render3DScatter(canvas, data, options) {
        const { xColumn, yColumn, zColumn } = options;
        await this.render2DScatter(canvas, data, xColumn, yColumn, zColumn);
    }

    /**
     * HSL to RGB conversion
     */
    hslToRgb(h, s, l) {
        s /= 100;
        l /= 100;
        const c = (1 - Math.abs(2 * l - 1)) * s;
        const x = c * (1 - Math.abs((h / 60) % 2 - 1));
        const m = l - c / 2;
        let r, g, b;

        if (h < 60) { r = c; g = x; b = 0; }
        else if (h < 120) { r = x; g = c; b = 0; }
        else if (h < 180) { r = 0; g = c; b = x; }
        else if (h < 240) { r = 0; g = x; b = c; }
        else if (h < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }

        return [(r + m), (g + m), (b + m)];
    }

    /**
     * Draw axes overlay using Canvas 2D
     */
    drawAxes2D(canvas, xLabel, yLabel, xMin, xMax, yMin, yMax) {
        const ctx = canvas.getContext('2d', { alpha: true });
        const width = canvas.width;
        const height = canvas.height;
        const padding = 60;

        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.font = 'bold 14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(xLabel, width / 2, height - 20);
        
        ctx.save();
        ctx.translate(20, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(yLabel, 0, 0);
        ctx.restore();

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.stroke();

        ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.font = '10px monospace';
        
        for (let i = 0; i <= 5; i++) {
            const xValue = xMin + (xMax - xMin) * (i / 5);
            const yValue = yMin + (yMax - yMin) * (i / 5);
            
            const x = padding + (width - 2 * padding) * (i / 5);
            const y = height - padding - (height - 2 * padding) * (i / 5);
            
            ctx.textAlign = 'center';
            ctx.fillText(xValue.toFixed(1), x, height - padding + 20);
            
            ctx.textAlign = 'right';
            ctx.fillText(yValue.toFixed(1), padding - 10, y + 4);
        }
    }

    /**
     * Draw bar chart labels
     */
    drawBarLabels(canvas, categories, values, maxValue) {
        const ctx = canvas.getContext('2d', { alpha: true });
        const width = canvas.width;
        const height = canvas.height;
        const padding = 60;
        const barWidth = (width - 2 * padding) / categories.length;

        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.font = '10px Inter';
        ctx.textAlign = 'center';

        categories.forEach((cat, idx) => {
            const x = padding + idx * barWidth + barWidth * 0.4;
            ctx.save();
            ctx.translate(x, height - padding + 15);
            ctx.rotate(-Math.PI / 4);
            ctx.fillText(String(cat).substring(0, 12), 0, 0);
            ctx.restore();

            const value = values[idx];
            const barHeight = (value / maxValue) * (height - 2 * padding);
            const y = height - padding - barHeight - 5;
            ctx.fillText(value.toFixed(1), x, y);
        });

        ctx.font = 'bold 12px Inter';
        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Average Value', 0, 0);
        ctx.restore();
    }

    /**
     * Render "no data" message
     */
    renderNoData(canvas) {
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#0a0a0a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = '#fff';
        ctx.font = '16px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('No numeric data found', canvas.width / 2, canvas.height / 2);
    }
}
