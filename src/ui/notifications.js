/**
 * Notification System
 * 
 * Toast notifications for user feedback
 */

export class NotificationManager {
    constructor() {
        this.container = null;
        this.initContainer();
    }

    /**
     * Initialize notification container
     */
    initContainer() {
        this.container = document.createElement('div');
        this.container.id = 'notification-container';
        this.container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 10px;
        `;
        document.body.appendChild(this.container);
    }

    /**
     * Show notification
     * @param {string} message - Notification message
     * @param {string} type - Notification type: success, error, warn, info
     * @param {number} duration - Auto-close duration in ms (0 = no auto-close)
     */
    show(message, type = 'info', duration = 4000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        
        const colors = {
            success: '#34B27B',
            error: '#ff4444',
            warn: '#ffaa00',
            info: '#4ECDC4'
        };
        
        const icons = {
            success: '✓',
            error: '✕',
            warn: '⚠',
            info: 'ℹ'
        };
        
        notification.style.cssText = `
            background: #11181C;
            border: 1px solid ${colors[type]};
            border-left: 4px solid ${colors[type]};
            color: white;
            padding: 16px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            min-width: 300px;
            max-width: 500px;
            display: flex;
            align-items: center;
            gap: 12px;
            animation: slideIn 0.3s ease;
        `;
        
        notification.innerHTML = `
            <span style="font-size: 20px; color: ${colors[type]}">${icons[type]}</span>
            <span style="flex: 1; font-size: 14px;">${message}</span>
            <button onclick="this.parentElement.remove()" style="
                background: none;
                border: none;
                color: rgba(255,255,255,0.5);
                cursor: pointer;
                font-size: 18px;
                padding: 0;
                width: 20px;
                height: 20px;
            ">&times;</button>
        `;
        
        this.container.appendChild(notification);
        
        // Auto-close
        if (duration > 0) {
            setTimeout(() => {
                notification.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => notification.remove(), 300);
            }, duration);
        }
        
        return notification;
    }

    /**
     * Shorthand methods
     */
    success(message, duration) {
        return this.show(message, 'success', duration);
    }

    error(message, duration) {
        return this.show(message, 'error', duration);
    }

    warn(message, duration) {
        return this.show(message, 'warn', duration);
    }

    info(message, duration) {
        return this.show(message, 'info', duration);
    }
}

// Add animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
