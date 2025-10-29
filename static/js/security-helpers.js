/**
 * Security helper functions for safe DOM manipulation
 */

// Safe text insertion that escapes HTML
function safeText(selector, text) {
    $(selector).text(text);
}

// Safe HTML construction using jQuery
function safeHtml(selector, builder) {
    const container = $(selector);
    container.empty();
    if (typeof builder === 'function') {
        builder(container);
    }
}

// Escape HTML entities
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Safe append with text content
function safeAppend(container, tag, text, classes) {
    const element = $(`<${tag}>`);
    if (classes) {
        element.addClass(classes);
    }
    element.text(text);
    $(container).append(element);
}

// Export for use in other scripts
window.SecurityHelpers = {
    safeText,
    safeHtml,
    escapeHtml,
    safeAppend
};
