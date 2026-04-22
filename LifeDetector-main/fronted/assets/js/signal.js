/**
 * Lightweight reactive state manager (Antigravity Signals)
 */
class Signal {
    constructor(initialValue) {
        this._value = initialValue;
        this.subscribers = new Set();
    }

    get value() {
        return this._value;
    }

    set value(newValue) {
        if (this._value !== newValue) {
            this._value = newValue;
            this.notify();
        }
    }

    subscribe(callback) {
        this.subscribers.add(callback);
        callback(this._value); // Initial call
        return () => this.subscribers.delete(callback); // Unsubscribe
    }

    notify() {
        this.subscribers.forEach(callback => callback(this._value));
    }
}

window.Signal = Signal;
