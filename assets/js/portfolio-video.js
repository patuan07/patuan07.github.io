/*
 * Looping videos.
 *
 * The markup carries no autoplay attribute and preload="none", and that is
 * deliberate: setting display:none on a <video> does not stop it downloading
 * or playing, so a reader who has asked for reduced motion would still pay for
 * the bytes and the decoder. The preference has to be honoured before play()
 * is ever called, which means here rather than in CSS.
 *
 * Without this script the poster frame is what shows, on the card and in the
 * write-up alike. That is the intended fallback, not a broken state.
 *
 * Playback is also gated on visibility. A write-up can carry several clips, and
 * preload="none" only defers the fetch until play() is called — calling play()
 * on every video at load would start every download and every decoder at once,
 * for clips the reader has not scrolled to. So a clip loads when it comes into
 * view and pauses when it leaves, which also keeps a backgrounded tab from
 * decoding video nobody is looking at.
 *
 * Videos opt in with data-autoplay; the poster covers the gap between first
 * paint and the first decoded frame.
 */
(function () {
  'use strict';

  var query = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
  if (query && query.matches) return;

  var videos = Array.prototype.slice.call(document.querySelectorAll('video[data-autoplay]'));
  if (!videos.length) return;

  function play(video) {
    var started = video.play();

    // Rejects under a strict autoplay policy (iOS Low Power Mode, or a browser
    // that ignores the muted exemption). The poster stays up, which is fine.
    if (started && started.catch) started.catch(function () {});
  }

  // Without IntersectionObserver every clip starts at once, which is what this
  // script did before there was more than one per page. Still correct, just
  // less considerate.
  if (!('IntersectionObserver' in window)) {
    videos.forEach(play);
    return;
  }

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          play(entry.target);
        } else {
          entry.target.pause();
        }
      });
    },
    // Start a little before the clip scrolls in, so it is already running by
    // the time it is properly on screen.
    { rootMargin: '200px 0px' }
  );

  videos.forEach(function (video) {
    observer.observe(video);
  });
})();
