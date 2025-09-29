import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes

def pdf_viewer(pdf_file):
    """PDF viewer with page navigation, zoom, and dummy highlighting."""

    # Convert PDF to images only once
    if "pdf_images" not in st.session_state or st.session_state.get("pdf_file_name") != pdf_file.name:
        pdf_bytes = pdf_file.read()
        st.session_state.pdf_images = convert_from_bytes(pdf_bytes)
        st.session_state.total_pages = len(st.session_state.pdf_images)
        st.session_state.current_page = 0
        st.session_state.zoom_level = 100
        st.session_state.highlights = []
        st.session_state.pdf_file_name = pdf_file.name

    # Toolbar
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Previous") and st.session_state.current_page > 0:
            st.session_state.current_page -= 1

    with col2:
        if st.button("Next") and st.session_state.current_page < st.session_state.total_pages - 1:
            st.session_state.current_page += 1

    with col3:
        zoom = st.select_slider("Zoom", options=["50%", "75%", "100%", "125%", "150%"], value=f"{st.session_state.zoom_level}%")
        st.session_state.zoom_level = int(zoom[:-1])

    with col4:
        if st.button("Highlight"):
            st.session_state.highlights.append({
                "page": st.session_state.current_page,
                "text": f"Highlighted text on page {st.session_state.current_page + 1}",
                "color": "yellow"
            })

    with col5:
        if st.button("Clear Highlights"):
            st.session_state.highlights = []

    # Display page number
    st.markdown(f"### Page {st.session_state.current_page + 1} of {st.session_state.total_pages}")
    st.markdown("---")

    # Display PDF page as image
    zoom_factor = st.session_state.zoom_level / 100.0
    image = st.session_state.pdf_images[st.session_state.current_page]
    width = int(image.width * zoom_factor)
    st.image(image, width=width)

    # Display highlights for current page
    current_highlights = [h for h in st.session_state.highlights if h["page"] == st.session_state.current_page]
    if current_highlights:
        st.markdown("**Highlights:**")
        for hl in current_highlights:
            st.markdown(f'<span style="background-color:{hl["color"]}">{hl["text"]}</span>', unsafe_allow_html=True)

    st.markdown("---")
