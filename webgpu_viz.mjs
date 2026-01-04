/**
 * WebGPU-Accelerated Visualization Engine for GridDB
 * Supports 1M+ points with real-time interaction
 */

export class WebGPUViz {
    constructor(device) {
        this.device = device;
        this.canvases = new Map();
        this.pipelines = new Map();
        this.initialized = false;
    }

    async init() {
        if (this.initialized) return;
        await this.createPipelines();
        this.initialized = true;
    }

    async createPipelines() {
        // 3D Scatter Plot Pipeline
        await this.create3DScatterPipeline();
        
        // 2D Chart Pipelines
        await this.create2DLinePipeline();
        await this.create2DBarPipeline();
        await this.create2DScatterPipeline();
        
        // Compute Pipelines for Statistics
        await this.createStatsComputePipeline();
    }

    /**
     * 3D Scatter Plot with WebGPU
     * Handles millions of points efficiently
     */
    async create3DScatterPipeline() {
        const shaderCode = `
            struct VertexInput {
                @location(0) position: vec3f,
                @location(1) color: vec3f,
                @location(2) size: f32,
            }

            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) color: vec3f,
                @location(1) size: f32,
            }

            struct Uniforms {
                viewProjection: mat4x4f,
                rotation: mat4x4f,
                scale: f32,
                time: f32,
            }

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;

            @vertex
            fn vertexMain(input: VertexInput) -> VertexOutput {
                var output: VertexOutput;
                
                // Apply rotation and scale
                var pos = uniforms.rotation * vec4f(input.position, 1.0);
                pos = pos * uniforms.scale;
                
                output.position = uniforms.viewProjection * pos;
                output.color = input.color;
                output.size = input.size;
                
                return output;
            }

            @fragment
            fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
                // Circular point with smooth edges
                let pointCoord = input.position.xy;
                let dist = length(pointCoord - 0.5);
                let alpha = smoothstep(0.5, 0.4, dist);
                
                return vec4f(input.color, alpha);
            }
        `;

        const shaderModule = this.device.createShaderModule({
            code: shaderCode,
            label: '3D Scatter Shader'
        });

        const pipelineDescriptor = {
            label: '3D Scatter Pipeline',
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
                buffers: [{
                    arrayStride: 28, // 3 (position) + 3 (color) + 1 (size) = 7 floats * 4 bytes
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x3' },  // position
                        { shaderLocation: 1, offset: 12, format: 'float32x3' }, // color
                        { shaderLocation: 2, offset: 24, format: 'float32' },   // size
                    ]
                }]
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat(),
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                        }
                    }
                }]
            },
            primitive: {
                topology: 'point-list',
            },
            depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: true,
                depthCompare: 'less',
            }
        };

        this.pipelines.set('3d-scatter', await this.device.createRenderPipelineAsync(pipelineDescriptor));
    }

    /**
     * 2D Line Chart with WebGPU
     */
    async create2DLinePipeline() {
        const shaderCode = `
            struct VertexInput {
                @location(0) position: vec2f,
                @location(1) color: vec3f,
            }

            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) color: vec3f,
            }

            struct Uniforms {
                transform: mat3x3f,
                aspectRatio: f32,
            }

            @group(0) @binding(0) var<uniform> uniforms: Uniforms;

            @vertex
            fn vertexMain(input: VertexInput) -> VertexOutput {
                var output: VertexOutput;
                
                let pos = uniforms.transform * vec3f(input.position, 1.0);
                output.position = vec4f(pos.xy, 0.0, 1.0);
                output.color = input.color;
                
                return output;
            }

            @fragment
            fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
                return vec4f(input.color, 1.0);
            }
        `;

        const shaderModule = this.device.createShaderModule({ code: shaderCode });

        this.pipelines.set('2d-line', await this.device.createRenderPipelineAsync({
            label: '2D Line Pipeline',
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
                buffers: [{
                    arrayStride: 20, // 2 (position) + 3 (color) = 5 floats
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x3' },
                    ]
                }]
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
            },
            primitive: {
                topology: 'line-strip',
            }
        }));
    }

    /**
     * 2D Bar Chart with WebGPU
     */
    async create2DBarPipeline() {
        const shaderCode = `
            struct VertexInput {
                @location(0) position: vec2f,
                @location(1) color: vec3f,
            }

            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) color: vec3f,
            }

            @vertex
            fn vertexMain(input: VertexInput) -> VertexOutput {
                var output: VertexOutput;
                output.position = vec4f(input.position, 0.0, 1.0);
                output.color = input.color;
                return output;
            }

            @fragment
            fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
                return vec4f(input.color, 1.0);
            }
        `;

        const shaderModule = this.device.createShaderModule({ code: shaderCode });

        this.pipelines.set('2d-bar', await this.device.createRenderPipelineAsync({
            label: '2D Bar Pipeline',
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
                buffers: [{
                    arrayStride: 20,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x3' },
                    ]
                }]
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
            },
            primitive: {
                topology: 'triangle-list',
            }
        }));
    }

    /**
     * 2D Scatter Plot with WebGPU
     */
    async create2DScatterPipeline() {
        const shaderCode = `
            struct VertexInput {
                @location(0) position: vec2f,
                @location(1) color: vec3f,
                @location(2) size: f32,
            }

            struct VertexOutput {
                @builtin(position) position: vec4f,
                @location(0) color: vec3f,
            }

            @vertex
            fn vertexMain(input: VertexInput) -> VertexOutput {
                var output: VertexOutput;
                output.position = vec4f(input.position, 0.0, 1.0);
                output.color = input.color;
                return output;
            }

            @fragment
            fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
                return vec4f(input.color, 0.9);
            }
        `;

        const shaderModule = this.device.createShaderModule({ 
            code: shaderCode,
            label: '2D Scatter Shader'
        });

        this.pipelines.set('2d-scatter', await this.device.createRenderPipelineAsync({
            label: '2D Scatter Pipeline',
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vertexMain',
                buffers: [{
                    arrayStride: 24, // 2 + 3 + 1 = 6 floats
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x3' },
                        { shaderLocation: 2, offset: 20, format: 'float32' },
                    ]
                }]
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragmentMain',
                targets: [{
                    format: navigator.gpu.getPreferredCanvasFormat(),
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
                        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }
                    }
                }]
            },
            primitive: {
                topology: 'triangle-list',
            }
        }));
    }

    /**
     * Compute Pipeline for Statistics (min, max, avg, histogram)
     */
    async createStatsComputePipeline() {
        const shaderCode = `
            struct Stats {
                min: atomic<u32>,
                max: atomic<u32>,
                sum: atomic<u32>,
                count: atomic<u32>,
            }

            @group(0) @binding(0) var<storage, read> data: array<f32>;
            @group(0) @binding(1) var<storage, read_write> stats: Stats;

            @compute @workgroup_size(256)
            fn computeStats(@builtin(global_invocation_id) id: vec3u) {
                let index = id.x;
                if (index >= arrayLength(&data)) {
                    return;
                }

                let value = data[index];
                let valueAsUint = bitcast<u32>(value);
                
                // Atomic operations for parallel reduction
                atomicMax(&stats.max, valueAsUint);
                atomicMin(&stats.min, valueAsUint);
                atomicAdd(&stats.count, 1u);
                
                // Note: sum needs special handling for floats
                // For now, we'll compute sum on CPU
            }
        `;

        const shaderModule = this.device.createShaderModule({ 
            code: shaderCode,
            label: 'Statistics Compute Shader'
        });

        this.pipelines.set('compute-stats', await this.device.createComputePipelineAsync({
            label: 'Statistics Compute Pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'computeStats',
            }
        }));
    }

    /**
     * Render 3D Scatter Plot
     */
    async render3DScatter(canvas, data, options = {}) {
        const {
            xColumn,
            yColumn,
            zColumn,
            colorColumn,
            rotation = { x: 0, y: 0, z: 0 },
            scale = 1.0
        } = options;

        const context = canvas.getContext('webgpu');
        context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
            alphaMode: 'premultiplied'
        });

        // Prepare vertex data
        const vertices = this.prepare3DScatterData(data, xColumn, yColumn, zColumn, colorColumn);
        
        // Create vertex buffer
        const vertexBuffer = this.device.createBuffer({
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(vertexBuffer, 0, vertices);

        // Create uniforms
        const uniformBuffer = this.createUniformBuffer(rotation, scale);

        // Render
        const commandEncoder = this.device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();
        
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.02, g: 0.02, b: 0.02, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });

        const pipeline = this.pipelines.get('3d-scatter');
        renderPass.setPipeline(pipeline);
        renderPass.setVertexBuffer(0, vertexBuffer);
        renderPass.draw(data.length);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);

        return { vertexBuffer, uniformBuffer };
    }

    /**
     * Prepare 3D scatter plot data
     */
    prepare3DScatterData(data, xCol, yCol, zCol, colorCol) {
        const vertices = new Float32Array(data.length * 7); // position(3) + color(3) + size(1)

        // Normalize data
        const xValues = data.map(d => parseFloat(d[xCol]) || 0);
        const yValues = data.map(d => parseFloat(d[yCol]) || 0);
        const zValues = data.map(d => parseFloat(d[zCol]) || 0);

        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);
        const zMin = Math.min(...zValues);
        const zMax = Math.max(...zValues);

        for (let i = 0; i < data.length; i++) {
            const baseIdx = i * 7;

            // Normalize positions to [-1, 1]
            vertices[baseIdx + 0] = 2 * (xValues[i] - xMin) / (xMax - xMin) - 1;
            vertices[baseIdx + 1] = 2 * (yValues[i] - yMin) / (yMax - yMin) - 1;
            vertices[baseIdx + 2] = 2 * (zValues[i] - zMin) / (zMax - zMin) - 1;

            // Color (gradient based on Z or category)
            const colorValue = colorCol ? this.getColorForValue(data[i][colorCol], i, data.length) : [0.2, 0.7, 0.5];
            vertices[baseIdx + 3] = colorValue[0];
            vertices[baseIdx + 4] = colorValue[1];
            vertices[baseIdx + 5] = colorValue[2];

            // Point size
            vertices[baseIdx + 6] = 5.0;
        }

        return vertices;
    }

    /**
     * Get color for data value
     */
    getColorForValue(value, index, total) {
        // If numeric, use gradient
        if (typeof value === 'number') {
            const t = value / 100; // Normalize
            return [
                0.2 + t * 0.5,
                0.7 - t * 0.3,
                0.5 + t * 0.3
            ];
        }
        
        // If categorical, use distinct colors
        const hue = (index / total) * 360;
        return this.hslToRgb(hue, 0.7, 0.6);
    }

    /**
     * HSL to RGB conversion
     */
    hslToRgb(h, s, l) {
        h = h / 360;
        const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
        const p = 2 * l - q;
        const r = this.hueToRgb(p, q, h + 1/3);
        const g = this.hueToRgb(p, q, h);
        const b = this.hueToRgb(p, q, h - 1/3);
        return [r, g, b];
    }

    hueToRgb(p, q, t) {
        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1/6) return p + (q - p) * 6 * t;
        if (t < 1/2) return q;
        if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
        return p;
    }

    /**
     * Create uniform buffer for transformations
     */
    createUniformBuffer(rotation, scale) {
        // 4x4 matrix + rotation matrix + scale + time = 32 floats
        const uniforms = new Float32Array(64);
        
        // View-projection matrix (identity for now)
        uniforms[0] = 1; uniforms[5] = 1; uniforms[10] = 1; uniforms[15] = 1;

        // Create rotation matrices
        const cos = Math.cos(rotation.y);
        const sin = Math.sin(rotation.y);
        uniforms[16] = cos; uniforms[17] = 0; uniforms[18] = sin;
        uniforms[20] = 0; uniforms[21] = 1; uniforms[22] = 0;
        uniforms[24] = -sin; uniforms[25] = 0; uniforms[26] = cos;

        // Scale
        uniforms[32] = scale;

        const buffer = this.device.createBuffer({
            size: uniforms.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(buffer, 0, uniforms);

        return buffer;
    }

    /**
     * Render 2D Scatter Plot
     */
    async render2DScatter(canvas, data, xCol, yCol, colorCol) {
        console.log('render2DScatter called:', { 
            dataLength: data.length, 
            xCol, 
            yCol, 
            colorCol,
            canvasSize: `${canvas.width}x${canvas.height}`
        });
        
        const context = canvas.getContext('webgpu');
        if (!context) {
            throw new Error('Failed to get WebGPU context from canvas');
        }
        
        context.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
        });

        // Prepare data - create quads for each point
        const vertices = this.prepare2DScatterDataAsQuads(data, xCol, yCol, colorCol);
        console.log('Prepared vertices:', vertices.length, 'floats');

        const vertexBuffer = this.device.createBuffer({
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(vertexBuffer, 0, vertices);

        // Render
        const commandEncoder = this.device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });

        const pipeline = this.pipelines.get('2d-scatter');
        renderPass.setPipeline(pipeline);
        renderPass.setVertexBuffer(0, vertexBuffer);
        renderPass.draw(data.length * 6); // 6 vertices per quad (2 triangles)
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
        
        console.log('✅ 2D scatter render complete');
    }

    prepare2DScatterDataAsQuads(data, xCol, yCol, colorCol) {
        // Each point becomes a quad (2 triangles = 6 vertices)
        const vertices = new Float32Array(data.length * 6 * 6); // 6 vertices * 6 floats per vertex

        const xValues = data.map(d => parseFloat(d[xCol]) || 0);
        const yValues = data.map(d => parseFloat(d[yCol]) || 0);

        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);
        
        console.log('Data ranges:', {
            xCol, xMin, xMax,
            yCol, yMin, yMax,
            points: data.length
        });

        const pointSize = 0.01; // Size in normalized coordinates

        for (let i = 0; i < data.length; i++) {
            const baseIdx = i * 36; // 6 vertices * 6 floats

            // Normalize position
            const x = 2 * (xValues[i] - xMin) / (xMax - xMin) - 1;
            const y = 2 * (yValues[i] - yMin) / (yMax - yMin) - 1;

            const color = this.getColorForValue(colorCol ? data[i][colorCol] : i, i, data.length);

            // Create quad (2 triangles)
            // Triangle 1
            // Vertex 0 (bottom-left)
            vertices[baseIdx + 0] = x - pointSize;
            vertices[baseIdx + 1] = y - pointSize;
            vertices[baseIdx + 2] = color[0];
            vertices[baseIdx + 3] = color[1];
            vertices[baseIdx + 4] = color[2];
            vertices[baseIdx + 5] = 1.0;

            // Vertex 1 (bottom-right)
            vertices[baseIdx + 6] = x + pointSize;
            vertices[baseIdx + 7] = y - pointSize;
            vertices[baseIdx + 8] = color[0];
            vertices[baseIdx + 9] = color[1];
            vertices[baseIdx + 10] = color[2];
            vertices[baseIdx + 11] = 1.0;

            // Vertex 2 (top-left)
            vertices[baseIdx + 12] = x - pointSize;
            vertices[baseIdx + 13] = y + pointSize;
            vertices[baseIdx + 14] = color[0];
            vertices[baseIdx + 15] = color[1];
            vertices[baseIdx + 16] = color[2];
            vertices[baseIdx + 17] = 1.0;

            // Triangle 2
            // Vertex 3 (top-left)
            vertices[baseIdx + 18] = x - pointSize;
            vertices[baseIdx + 19] = y + pointSize;
            vertices[baseIdx + 20] = color[0];
            vertices[baseIdx + 21] = color[1];
            vertices[baseIdx + 22] = color[2];
            vertices[baseIdx + 23] = 1.0;

            // Vertex 4 (bottom-right)
            vertices[baseIdx + 24] = x + pointSize;
            vertices[baseIdx + 25] = y - pointSize;
            vertices[baseIdx + 26] = color[0];
            vertices[baseIdx + 27] = color[1];
            vertices[baseIdx + 28] = color[2];
            vertices[baseIdx + 29] = 1.0;

            // Vertex 5 (top-right)
            vertices[baseIdx + 30] = x + pointSize;
            vertices[baseIdx + 31] = y + pointSize;
            vertices[baseIdx + 32] = color[0];
            vertices[baseIdx + 33] = color[1];
            vertices[baseIdx + 34] = color[2];
            vertices[baseIdx + 35] = 1.0;
        }
        
        console.log('First point:', {
            x: vertices[0],
            y: vertices[1],
            color: [vertices[2], vertices[3], vertices[4]]
        });

        return vertices;
    }

    prepare2DScatterData(data, xCol, yCol, colorCol) {
        const vertices = new Float32Array(data.length * 6); // x, y, r, g, b, size

        const xValues = data.map(d => parseFloat(d[xCol]) || 0);
        const yValues = data.map(d => parseFloat(d[yCol]) || 0);

        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);

        for (let i = 0; i < data.length; i++) {
            const baseIdx = i * 6;

            vertices[baseIdx + 0] = 2 * (xValues[i] - xMin) / (xMax - xMin) - 1;
            vertices[baseIdx + 1] = 2 * (yValues[i] - yMin) / (yMax - yMin) - 1;

            const color = this.getColorForValue(colorCol ? data[i][colorCol] : i, i, data.length);
            vertices[baseIdx + 2] = color[0];
            vertices[baseIdx + 3] = color[1];
            vertices[baseIdx + 4] = color[2];

            vertices[baseIdx + 5] = 3.0;
        }

        return vertices;
    }

    /**
     * Compute statistics using GPU (simplified version - uses CPU for now)
     * Full GPU version with parallel reduction coming soon
     */
    async computeStats(data, column) {
        const values = data.map(d => parseFloat(d[column]) || 0).filter(v => !isNaN(v));
        
        if (values.length === 0) {
            return { min: 0, max: 0, avg: 0, count: 0 };
        }

        // CPU computation (fast enough for most cases)
        // TODO: Implement GPU parallel reduction for 10M+ rows
        const min = Math.min(...values);
        const max = Math.max(...values);
        const sum = values.reduce((a, b) => a + b, 0);
        const avg = sum / values.length;

        return {
            min,
            max,
            avg,
            sum,
            count: values.length,
            median: this.calculateMedian(values),
            stdDev: this.calculateStdDev(values, avg)
        };
    }

    /**
     * Calculate median
     */
    calculateMedian(values) {
        const sorted = [...values].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0
            ? (sorted[mid - 1] + sorted[mid]) / 2
            : sorted[mid];
    }

    /**
     * Calculate standard deviation
     */
    calculateStdDev(values, avg) {
        const squaredDiffs = values.map(v => Math.pow(v - avg, 2));
        const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
        return Math.sqrt(avgSquaredDiff);
    }
}
