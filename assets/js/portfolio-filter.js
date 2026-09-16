/*
 * Topic filter for /projects/.
 *
 * Progressive enhancement: the filter row is rendered hidden and revealed
 * here, so a reader without JavaScript sees the full list and no buttons that
 * do nothing. Card tags come from `data-tags` on .list__item (pipe-separated,
 * lowercased by the template).
 */
(function () {
  'use strict';

  var row = document.querySelector('.portfolio-filter');
  var grid = document.getElementById('portfolio-grid');
  if (!row || !grid) return;

  var empty = document.getElementById('portfolio-empty');
  var items = Array.prototype.slice.call(grid.querySelectorAll('.list__item'));
  var buttons = Array.prototype.slice.call(row.querySelectorAll('button[data-filter]'));

  function apply(tag) {
    var shown = 0;

    items.forEach(function (item) {
      var tags = (item.getAttribute('data-tags') || '').split('|');
      var match = tag === 'all' || tags.indexOf(tag) !== -1;
      item.hidden = !match;
      if (match) shown += 1;
    });

    buttons.forEach(function (button) {
      var active = button.getAttribute('data-filter') === tag;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });

    if (empty) empty.hidden = shown !== 0;
  }

  row.addEventListener('click', function (event) {
    var button = event.target.closest('button[data-filter]');
    if (button) apply(button.getAttribute('data-filter'));
  });

  // Allow linking straight to a filtered view: /projects/?topic=ros%202
  var requested = new URLSearchParams(window.location.search).get('topic');
  if (requested && buttons.some(function (b) { return b.getAttribute('data-filter') === requested; })) {
    apply(requested);
  }

  row.hidden = false;
})();
