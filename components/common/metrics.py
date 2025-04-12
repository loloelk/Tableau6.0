"""Common metrics and display components for the dashboard."""
import streamlit as st
import pandas as pd

def custom_progress_bar(
    progress_value, 
    height="10px",
    bg_color="#f5f5f5",
    fill_color="linear-gradient(90deg, #2196f3, #64b5f6)", 
    text_before="",
    text_after="",
    include_percentage=True,
    container_styles="margin: 8px 0 15px 0;",
    animate=True
):
    """Create a custom styled progress bar that doesn't use Streamlit's default.
    
    Args:
        progress_value: Value between 0 and 1 representing progress percentage
        height: Height of the progress bar
        bg_color: Background color for unfilled portion
        fill_color: Color for the filled portion (can be gradient)
        text_before: Text to display before the progress bar
        text_after: Text to display after the progress bar
        include_percentage: Whether to show percentage text
        container_styles: Additional CSS styles for the container
        animate: Whether to animate the progress bar filling
    """
    # Ensure progress value is between 0 and 1
    progress_value = min(max(progress_value, 0), 1)
    
    # Format percentage for display
    percentage = f"{int(progress_value * 100)}%"
    
    # Build the HTML/CSS for the progress bar
    animation = "transition: width 1s ease-in-out;" if animate else ""
    
    # Create a more visible progress bar with guaranteed display
    progress_html = f"""
    <div style="width: 100%; {container_styles} display: block !important;">
        {f"<div style='margin-bottom: 5px; font-size: 0.9em; color: #555;'>{text_before}</div>" if text_before else ""}
        <div style="display: flex; align-items: center; margin-bottom: 5px; width: 100%;">
            <div style="flex-grow: 1; position: relative; background-color: {bg_color}; height: {height}; border-radius: 10px; overflow: hidden;">
                <div style="position: absolute; top: 0; left: 0; height: 100%; width: {percentage}; background: {fill_color}; border-radius: {height}; {animation}"></div>
            </div>
            {f"<div style='margin-left: 10px; font-weight: 500; color: #555; font-size: 0.9em;'>{percentage}</div>" if include_percentage else ""}
        </div>
        {f"<div style='font-size: 0.85em; color: #777;'>{text_after}</div>" if text_after else ""}
    </div>
    """
    
    # Use a container to ensure proper rendering
    container = st.container()
    container.markdown(progress_html, unsafe_allow_html=True)

def milestone_progress(
    current_milestone, 
    milestones, 
    completed_color="#2196f3", 
    active_color="#3f51b5",
    inactive_color="#e0e0e0"
):
    """Display milestones with visual indicators of progress.
    
    Args:
        current_milestone: Index of the current milestone (0-based)
        milestones: List of milestone labels
        completed_color: Color for completed milestones
        active_color: Color for current milestone
        inactive_color: Color for future milestones
    """
    # Generate HTML for milestone indicators
    milestone_count = len(milestones)
    
    # Create a container for milestones
    milestones_container = st.container()
    
    # First create a flex container div
    milestones_container.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin: 20px 0; width: 100%;">
    """, unsafe_allow_html=True)
    
    # Create columns for milestones
    cols = st.columns(milestone_count)
    
    # Add milestone markers in each column
    for i, (col, milestone) in enumerate(zip(cols, milestones)):
        with col:
            # Determine the status of this milestone
            if i < current_milestone:
                # Completed milestone
                color = completed_color
                opacity = "1"
                font_weight = "normal"
                check_mark = "✓"
                box_shadow = ""
            elif i == current_milestone:
                # Current milestone
                color = active_color
                opacity = "1"
                font_weight = "bold"
                check_mark = ""
                box_shadow = "box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.2);"
            else:
                # Future milestone
                color = inactive_color
                opacity = "0.5"
                font_weight = "normal"
                check_mark = ""
                box_shadow = ""
            
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: center; width: 100%;">
                <div style="
                    width: 16px;
                    height: 16px;
                    border-radius: 50%;
                    background-color: {color};
                    margin-bottom: 5px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    color: white;
                    font-size: 10px;
                    {box_shadow}
                ">{check_mark}</div>
                <div style="text-align: center; font-size: 0.8em; opacity: {opacity}; font-weight: {font_weight};">
                    {milestone}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Close the flex container
    milestones_container.markdown("</div>", unsafe_allow_html=True)