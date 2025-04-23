from crewai import Agent, Task, Crew, Process
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from langchain_openai import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from crewai.tools import BaseTool
from pytube import YouTube
import cv2
import os
import datetime
import numpy as np
import re
import tempfile
from typing import List, Dict, Optional
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def extract_youtube_id(youtube_url):
    """Extract the video ID from a YouTube URL"""
    youtube_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
    if youtube_id_match:
        return youtube_id_match.group(1)
    return None

class TranscriptExtractionTool(BaseTool):
    name: str = "Transcript Extractor"
    description: str = "Extracts transcripts from YouTube videos with timestamps"
    
    def _run(self, video_url: str) -> list:
        """Extract transcript from YouTube URL or ID"""
        try:
            # Handle both full URLs and direct video IDs
            if "youtube.com" in video_url or "youtu.be" in video_url:
                video_id = extract_youtube_id(video_url)
            else:
                video_id = video_url
                
            if not video_id:
                return f"Error: Could not extract video ID from {video_url}"
                
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return [{
                "text": entry['text'],
                "start": entry['start'],
                "duration": entry['duration']
            } for entry in transcript]
        except Exception as error:
            return f"Error fetching transcript: {str(error)}"

class DiagramExtractionTool(BaseTool):
    name: str = "Diagram Extractor"
    description: str = (
        "Downloads YouTube video and identifies diagrams from frames using OpenCV."
    )
    output_dir: str = Field(default="diagrams", description="Output directory for extracted diagrams")
    frame_interval: int = Field(default=5, description="Frame sampling interval for efficiency")

    def _run(self, video_url: str) -> List[Dict]:
        """Downloads YouTube video and extracts diagrams"""
        import cv2
        import os
        import numpy as np
        import tempfile
        from pytube import YouTube
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        diagrams = []
        temp_video_file = None
        
        try:
            # Extract video ID
            if "youtube.com" in video_url or "youtu.be" in video_url:
                video_id = extract_youtube_id(video_url)
            else:
                video_id = video_url
                
            if not video_id:
                return [{"error": f"Could not extract video ID from {video_url}"}]
            
            # Download the video to a temporary file
            print(f"Downloading YouTube video: {video_url}")
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            stream = yt.streams.filter(progressive=True, file_extension="mp4").first()
            
            # Create a temporary file
            temp_fd, temp_video_file = tempfile.mkstemp(suffix=".mp4")
            os.close(temp_fd)
            
            # Download to the temporary file
            stream.download(filename=temp_video_file)
            print(f"Downloaded video to: {temp_video_file}")
            
            # Now process the video file
            cap = cv2.VideoCapture(temp_video_file)
            if not cap.isOpened():
                return [{"error": "Failed to open video capture"}]
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            prev_frame = None
            frame_count = 0

            print("Processing video frames for diagrams...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_count % self.frame_interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Skip first frame comparison
                    if prev_frame is not None:
                        # Calculate difference between frames
                        diff = cv2.absdiff(prev_frame, gray)
                        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                        
                        # If significant change is detected between frames, possibly a diagram or slide change
                        if np.mean(thresh) > 5:
                            timestamp = frame_count / fps
                            filename = os.path.join(self.output_dir, f"diagram_{frame_count}.jpg")
                            cv2.imwrite(filename, frame)
                            
                            # Convert timestamp to minutes:seconds format
                            mins = int(timestamp // 60)
                            secs = int(timestamp % 60)
                            timestamp_str = f"{mins}:{secs:02d}"
                            
                            diagrams.append({
                                "frame": frame_count,
                                "timestamp": timestamp,
                                "timestamp_str": timestamp_str,
                                "path": filename
                            })
                            print(f"Diagram extracted at {timestamp_str}")
                    
                    prev_frame = gray
                frame_count += 1

            # If we couldn't detect any diagrams, return a message
            if not diagrams:
                return [{"note": "No significant visual changes detected for diagram extraction"}]
                
            return diagrams
            
        except Exception as e:
            return [{"error": f"Error in diagram extraction: {str(e)}"}]
        finally:
            # Cleanup temporary file
            if temp_video_file and os.path.exists(temp_video_file):
                os.remove(temp_video_file)
                print(f"Removed temporary video file: {temp_video_file}")

def integrate_lecture_notes(generated_notes, pdf_paths):
    """Enhance generated notes with content from lecture materials"""
    if not pdf_paths:
        return generated_notes
        
    # Load lecture notes from PDFs
    lecture_notes_source = CrewDoclingSource(
        files=pdf_paths,
        metadata={"type": "lecture_notes"}
    )
    
    # Create a tool for searching the lecture notes
    lecture_notes_tool = lecture_notes_source.as_tool()
    
    # Define integration agent
    integration_agent = Agent(
        role="Lecture Notes Integrator",
        goal="Enhance generated notes with relevant information from lecture materials",
        backstory="An expert at finding and integrating complementary information",
        tools=[lecture_notes_tool],
        llm=ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.0,
        ),
        verbose=True
    )
    
    # Define task for integration
    integration_task = Task(
        description="""
        Enhance the generated notes with relevant information from the lecture materials.
        Look for:
        1. Additional explanations for key concepts
        2. Examples or case studies not mentioned in the transcript
        3. Relevant formulas or technical details
        4. References to additional resources
        
        Insert this information at appropriate points in the notes, clearly indicating
        that it comes from supplementary materials.
        """,
        expected_output="Enhanced notes with integrated information from lecture materials",
        agent=integration_agent
    )
    
    # Create a crew just for this task
    integration_crew = Crew(
        agents=[integration_agent],
        tasks=[integration_task],
        verbose=True
    )
    
    # Run the integration
    enhanced_notes = integration_crew.kickoff(inputs={"generated_notes": generated_notes})
    
    return enhanced_notes

def generate_notes_from_video(video_url, lecture_notes_paths=None, output_dir="output"):
    """Main function to generate lecture notes from a YouTube video"""
    os.makedirs(output_dir, exist_ok=True)
    
    video_id = extract_youtube_id(video_url)
    if not video_id:
        print(f"Error: Could not extract video ID from {video_url}")
        return None
    
    # Initialize agents
    transcript_analyzer = Agent(
        role="Transcript Analyzer",
        goal="Create comprehensive and detailed notes by analyzing the content, focusing on concepts and their relationships",
        backstory="An expert educator skilled in transforming lecture content into clear, detailed, and well-organized educational material",
        tools=[TranscriptExtractionTool()],
        llm=ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.0,
        ),
        verbose=True
    )

    diagram_extractor = Agent(
        role="Diagram Extractor",
        goal="Extract and contextualize visual elements to enhance understanding of concepts",
        backstory="A visual learning expert who seamlessly integrates diagrams and illustrations into educational content",
        tools=[DiagramExtractionTool()],
        llm=ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.0,
        ),
        verbose=True
    )

    note_formatter = Agent(
        role="Note Formatter",
        goal="Transform analyzed content into detailed, well-structured educational notes that prioritize understanding",
        backstory="""An expert in educational content organization who:
        - Creates detailed concept explanations
        - Develops clear conceptual hierarchies
        - Provides comprehensive examples and applications
        - Integrates visual elements meaningfully
        - Ensures deep understanding through proper structuring and explanation""",
        llm=ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.0,
        ),
        verbose=True
    )
    
    # Define tasks
    transcript_analysis_task = Task(
        description=f"""
        Create detailed educational notes from the video content at {video_url} (ID: {video_id}).
        Focus on:
        1. Comprehensive explanation of main concepts
        2. Detailed breakdown of complex ideas
        3. Real-world applications and examples
        4. Interconnections between different concepts
        5. Thorough explanations of processes and methodologies
        
        Organize content to build understanding progressively, focusing on depth and clarity.
        Avoid timestamp-based structuring, instead focus on logical concept progression.
        """,
        expected_output="Detailed, concept-focused analysis of the lecture content",
        agent=transcript_analyzer
    )
    
    diagram_extraction_task = Task(
        description=f"""
        Extract and analyze visual elements from the video at {video_url} that:
        1. Demonstrate key concepts in detail
        2. Illustrate complex processes
        3. Show relationships between ideas
        4. Support theoretical explanations
        
        For each visual element:
        - Provide detailed explanation of its significance
        - Explain how it relates to the main concepts
        - Describe its role in understanding the topic
        """,
        expected_output="Analyzed visual elements with detailed explanations of their educational significance",
        agent=diagram_extractor,
        dependencies=[transcript_analysis_task]
    )
    
    note_formatting_task = Task(
        description="""
        Create comprehensive educational notes that:
        1. Provide in-depth explanations of each concept
        2. Include detailed examples and applications
        3. Offer thorough analysis of processes and methodologies
        4. Integrate visual elements with proper context and explanation
        
        Structure the content using:
        - Clear conceptual hierarchies
        - Detailed subsections for complex topics
        - Comprehensive examples and case studies
        - Proper integration of visual aids with explanations
        
        Formatting guidelines:
        - Use headers to show concept hierarchy
        - Include detailed bullet points for key ideas
        - Provide block quotes for important definitions
        - Create tables for comparing concepts
        - Use emphasis for crucial terms
        - Include code blocks for technical content
        
        Focus on creating an educational resource that provides deep understanding
        of the subject matter, avoiding timestamp-based organization.
        """,
        expected_output="Comprehensive, concept-focused educational notes in Markdown format",
        agent=note_formatter,
        dependencies=[diagram_extraction_task]
    )
    
    # Create and run the crew
    notes_crew = Crew(
        agents=[transcript_analyzer, diagram_extractor, note_formatter],
        tasks=[transcript_analysis_task, diagram_extraction_task, note_formatting_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Execute the workflow
    print(f"Starting note generation for video: {video_url}")
    generated_notes_output = notes_crew.kickoff(inputs={"video_url": video_url})
    
    # Extract string content from CrewOutput object
    if hasattr(generated_notes_output, "raw_output"):
        generated_notes = generated_notes_output.raw_output
    else:
        # If raw_output attribute doesn't exist, convert the object to string
        generated_notes = str(generated_notes_output)
    
    # Integrate lecture notes if available
    if lecture_notes_paths:
        print("Integrating lecture notes...")
        enhanced_notes_output = integrate_lecture_notes(generated_notes, lecture_notes_paths)
        
        # Extract string content
        if hasattr(enhanced_notes_output, "raw_output"):
            enhanced_notes = enhanced_notes_output.raw_output
        else:
            enhanced_notes = str(enhanced_notes_output)
    else:
        enhanced_notes = generated_notes
    
    # Save the notes
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/lecture_notes_{timestamp}.md"
    
    with open(output_path, "w") as f:
        f.write(enhanced_notes)
    
    print(f"Notes saved to {output_path}")
    return output_path

# Example usage
if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=rB83DpBJQsE&t=215s"
    lecture_notes = []  # Optional list of PDF paths for lecture notes
    notes_path = generate_notes_from_video(video_url, lecture_notes)
