from crewai import Agent, Task, Crew, Process
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from langchain_openai import ChatOpenAI
from crewai.tools import BaseTool
from dotenv import load_dotenv
from crewai.memory.memory import Memory
import cv2
import os
import re
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


class FileReadingTool(BaseTool):
    name: str = "FileReadingTool"
    description: str = "Reads a text file and returns its content."

    def _run(self, file_path: str) -> str:
        """Reads a text file and returns its content."""
        file_path = os.path.abspath(file_path)  # Ensure absolute path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path} (Checked path: {file_path})")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return content



class VideoProcessingTool(BaseTool):
    name: str = "VideoProcessingTool"
    description: str = "Extracts frames from a video at specific timestamps."

    def _run(self, video_path: str, timestamp: str, output_dir: str) -> str:
        """
        Extracts a frame from the video at the given timestamp and saves it as an image.
        
        Parameters:
            video_path (str): Path to the video file.
            timestamp (str): Timestamp in HH:MM:SS format.
            output_dir (str): Directory to save the extracted image.
        
        Returns:
            str: Path to the saved image file.
        """
        # Convert timestamp to seconds
        h, m, s = map(int, timestamp.split(':'))
        frame_time = h * 3600 + m * 60 + s

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Set the frame position
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(frame_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        success, frame = cap.read()
        if not success:
            raise ValueError(f"Could not read frame at {timestamp} in {video_path}")

        # Save the frame as an image
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"frame_{h:02d}_{m:02d}_{s:02d}.jpg")
        cv2.imwrite(output_file, frame)

        # Release resources
        cap.release()

        # Compute relative path for markdown: assume markdown is saved at "output/lecture_notes.md"
        project_root = os.getcwd()
        rel_path = os.path.relpath(output_file, project_root)
        return rel_path



class MarkdownFormattingTool(BaseTool):
    name: str = "MarkdownFormattingTool"
    description: str = "Formats content into markdown and saves it as a .md file."

    def _run(self, content: str, output_file: str) -> str:
        """
        Formats content into markdown and saves it as a .md file.
        
        Parameters:
            content (str): The text content to format.
            output_file (str): Path to save the markdown file.
        
        Returns:
            str: Path to the saved markdown file.
        """
        # Ensure proper markdown formatting
        formatted_content = self._format_markdown(content)

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(formatted_content)

        return output_file

    def _format_markdown(self, content: str) -> str:
        """
        Applies basic markdown formatting rules to ensure consistency.
        
        Parameters:
            content (str): The raw text content.
        
        Returns:
            str: Formatted markdown text.
        """
        # Ensure headings start with correct number of '#' symbols
        content = re.sub(r'^(#+)', r'\1 ', content, flags=re.MULTILINE)

        # Add spacing around headings for readability
        content = re.sub(r'(^#+ .+)', r'\n\1\n', content, flags=re.MULTILINE)

        # Ensure bullet points are properly formatted
        content = re.sub(r'^(\*|\-|\+)\s+', r'- ', content, flags=re.MULTILINE)

        return content


transcript_analyzer = Agent(
    role="Transcript Analysis Expert",
    goal="Create a comprehensive and well-structured outline from transcript content",
    backstory="""You are an experienced educational content analyst with expertise in 
    breaking down complex information into organized structures. Your specialty is creating 
    clear, hierarchical outlines that capture the essence of lectures and presentations.""",
    tools=[FileReadingTool()],
    llm=ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.2,
        ),
    verbose=True
)

diagram_identifier = Agent(
    role="Diagram References Specialist",
    goal="Identify all mentions of visual aids along with their timestamps",
    backstory="""You are a specialized analyst who excels at identifying references to visual 
    materials within spoken content. You're adept at noting precise timestamps for these 
    references to ensure visuals can be aligned with textual content.""",
    tools=[FileReadingTool()],
    llm=ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.4,
        ),
    verbose=True
)

diagram_extractor = Agent(
    role="Video Frame Extraction Specialist",
    goal="Extract high-quality image frames from videos at precise timestamps",
    backstory="""You are an expert in video processing with a background in multimedia 
    content management. Your specialty is extracting the perfect frame from videos when 
    diagrams or visual aids are being displayed.""",
    tools=[VideoProcessingTool()],
    llm=ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.5,
        ),
    verbose=True
)

context_adder = Agent(
    role="Educational Content Enrichment Specialist",
    goal="Integrate textual and visual elements to create comprehensive educational notes",
    backstory="""You specialize in creating rich, contextual educational materials by 
    combining outlines, explanatory text, and visual elements. You excel at expanding brief 
    points into detailed explanations while maintaining clarity.""",
    memory=Memory(memory_type="buffer", memory_config={"buffer_size": 15}, storage={}),
    llm=ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.5,
        ),
    verbose=True
)

notes_formatter = Agent(
    role="Educational Content Formatting Expert",
    goal="Transform comprehensive notes into well-formatted, accessible markdown documents",
    backstory="""You are a specialist in document formatting with deep knowledge of markdown, 
    educational content design, and information architecture. You understand how to structure 
    documents for maximum readability and learning effectiveness.""",
    tools=[MarkdownFormattingTool()],
    llm=ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.5,
        ),
    verbose=True
)

transcript_analysis_task = Task(
    description="""
    Analyze the provided transcript and create a detailed educational outline of the content.
    Using the Transcript at : {transcript_file}, identify the core main topics, subtopics, and technical concepts.
    Your analysis should focus on:
    1. Main themes and concepts
    2. Indepth explanation and examples
    3. Logical flow of information
    4. Key points and supporting details
    5. Any references to diagrams or visual aids
    6. Contextual information that enhances understanding
    7. Any other relevant details that can aid in creating educational notes
    8. Formulas and Technical Knowledge
    9. Identification of possible visual aids (only if they enhance the educational material)
    """,
    expected_output="""
    A detailed educational outline in markdown format with clear headings, subheadings, and comprehensive explanations.""",
   
    agent=transcript_analyzer
)


diagram_identification_task = Task(
    description="""
    Analyze the transcript to identify all mentions of relevant educational diagrams or visual aids. DO NOT FORCE DIAGRAMS
    
    Using the Transcript at : {transcript_file}, identify the timestamps and context for each diagram reference
    1. Identify the context in which each diagram is referenced
    For each identified reference:
    1. Note the exact timestamp in HH:MM:SS format
    2. Extract the context around the reference
    3. Identify the appropriate location in the outline where the diagram should be inserted and it should make sense
    """,
    expected_output="""
    A JSON-formatted list of diagram references with timestamps, context, and outline locations.
    """,
    agent=diagram_identifier,
    context=[transcript_analysis_task]
)

diagram_extraction_task = Task(
    description="""
    Using the list of diagram timestamps, extract the corresponding frames from the video file.
    Use the Following Video : {video_file}
    Save each extracted frame as an image file in a specified directory : {frames_dir}
    For each timestamp:
    1. Extract the frame from the video
    2. Save the image with a descriptive filename
    3. Create robust mapping between timestamps and image files so that the extracted images can be easily referenced in the notes
    """,
    expected_output="""
    A JSON-formatted list mapping timestamps to extracted image filenames.
    """,
    agent=diagram_extractor,
    context=[diagram_identification_task]
)

context_addition_task = Task(
    description="""
     Using the outline from the transcript analysis, expand each section into comprehensive educational notes.
    Using the extracted diagrams, integrate them into the notes at appropriate locations.
    You can use your knowledge to add context and explanations where necessary.
    
    For each section:
    1. Only integrate visual aids (if available and relevant) rather than blindly inserting all extracted frames
    2. Expand brief descriptions into detailed explanations
    3. Insert relevant diagrams at appropriate locations
    4. Add explanatory text around diagrams
    5. Ensure smooth transitions between topics
    6. Create proper Formulas and Technical Knowledge
    """,
    expected_output="""
    Comprehensive educational notes with detailed explanations, technical examples, and carefully integrated visuals (when relevant).
    """,
    agent=context_adder,
    context=[transcript_analysis_task, diagram_extraction_task]
)

notes_formatting_task = Task(
    description="""
    Transform the comprehensive educational notes into a polished markdown document.
    Your final document should:
    1. Use a consistent heading hierarchy for clear organization.
    2. Include a table of contents to facilitate navigation.
    3. Display metadata (e.g., creation date, subject) as needed.
    4. Present diagrams with descriptive captions and proper placement.
    5. Apply clean markdown styling for readability.
    6. Format formulas appropriately according to markdown rules.
    7. Maintain an academic tone with detailed explanations and examples.
    """,
    expected_output="""
    A fully formatted markdown file as a single string, ready to be saved under 'output/lecture_notes.md'.
    """,
    agent=notes_formatter,
    context=[context_addition_task],
    output_file="output/lecture_notes.md"
)


notes_creation_crew = Crew(
    agents=[
        transcript_analyzer,
        diagram_identifier,
        diagram_extractor,
        context_adder,
        notes_formatter
    ],
    tasks=[
        transcript_analysis_task,
        diagram_identification_task,
        diagram_extraction_task,
        context_addition_task,
        notes_formatting_task
    ],
    process=Process.sequential,
    verbose=True,
    output_log_file = True
    
)


if __name__ == "__main__":
    inputs = {
    "transcript_file": os.path.abspath("Data/Transcript/Video3Transcript.txt"),
    "video_file": os.path.abspath("Data/Video/Video3.mp4"),
    "frames_dir": os.path.abspath("Data/Frames"),
}
    output = notes_creation_crew.kickoff(inputs=inputs)  # Execute workflow
    # crew = Crew(output_log_file = True) 
    print("Workflow completed. Output saved to output/lecture_notes.md")
