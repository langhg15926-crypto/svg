(() => {
  const modal = document.getElementById('modal');
  const btnSettings = document.getElementById('btnSettings');
  const btnCloseModal = document.getElementById('btnCloseModal');

  if (!modal || !btnSettings || !btnCloseModal) {
    return;
  }

  const openModal = () => {
    modal.classList.add('show');
  };

  const closeModal = () => {
    modal.classList.remove('show');
  };

  btnSettings.addEventListener('click', openModal);
  btnCloseModal.addEventListener('click', closeModal);

  modal.addEventListener('click', (event) => {
    if (event.target === modal) {
      closeModal();
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && modal.classList.contains('show')) {
      closeModal();
    }
  });
})();
