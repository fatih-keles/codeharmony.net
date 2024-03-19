from flask import (
    render_template, request, after_this_request, send_file, redirect, url_for, flash
)
from pytube import YouTube
from urllib.parse import urlparse, parse_qs
import re
import tempfile
import os

from . import video_tools_bp

@video_tools_bp.route('/download')
def download():
    return render_template('video_tools/download.html', default_presets=[])

@video_tools_bp.route('/download/get-video-details', methods=['POST'])
def get_video_details():
    json_response = {
        "status": "success",
        "description": "Successfuly completed operation",
    }
    input_url = request.form.get("URLInput", "")
    if not is_url(input_url):
        return {
            "status": "error",
            "description": "Not a valid URL",
        }
    
    resource_type = detect_resource_type(input_url)
    if resource_type != "youtube":
        return {
            "status": "error",
            "description": "Not a supported resource type! Provide a valid YouTube video URL",
        }
    
    json_response["resource_type"] = resource_type
    video_url = input_url
    try:
        yt = YouTube(video_url)
        json_response["title"] = yt.title
        json_response["thumbnail_uri"] = yt.thumbnail_url
        streams = []
        for s in yt.streams.filter(file_extension='mp4', only_video=True):
            print(s)
            streams.append({
                "itag": s.itag,
                "mime_type": s.mime_type,
                "type": s.type,
                "resolution": s.resolution if s.type == "video" else None,
                "abr": s.abr if s.type == "audio" else None
            })
        json_response["streams"] = streams
    except Exception as error:
        print(f"Error: {error}")
        return {
            "status": "error",
            "description": f"Error: {error}",
        }
    return json_response

@video_tools_bp.route('/download/get-video-by-tag-id', methods=['POST'])
def download_video_by_tag():
    input_url = request.form.get("URLInput", "")
    if not is_url(input_url):
        flash('Not a valid URL', 'error')
        return redirect(url_for('video_tools.download'))
    
    resource_type = detect_resource_type(input_url)
    if resource_type != "youtube":
        flash('Not a supported resource type! Provide a valid YouTube video URL', 'error')
        return redirect(url_for('video_tools.download'))
    
    itag = request.form.get("resolution", "")
    if not itag:
        flash('No resolution selected', 'error')
        return redirect(url_for('video_tools.download'))
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as video_file:
        try:
            yt = YouTube(input_url)
            stream = yt.streams.get_by_itag(itag)
            stream.download(output_path = tempfile.gettempdir(), filename = video_file.name)
            print(f"YouTube video downloaded successfully at {tempfile.gettempdir()} and file name is {video_file.name}.")
            friendly_filename = yt.title + "-"+stream.resolution+".mp4"
            return send_file(video_file.name, as_attachment=True, download_name=friendly_filename)
        except Exception as error:
            print(f"Error while downloading the video: {error}")
            flash(f"Error while downloading the video: {error}", 'error')
            return redirect(url_for('video_tools.download'))
        finally:
            # Delete the temporary file
            os.remove(video_file.name)
            print("Temporary files deleted.")
    
def is_url(url):
    try:
        result = urlparse(url)
        # <scheme>://<netloc>/<path>;<params>?<query>#<fragment>
        # return all([result.scheme, result.netloc])
        return result.scheme.lower() in ['http', 'https']
    except ValueError:
        return False
    
def detect_resource_type(url):
    try:
        result = urlparse(url)
        # <scheme>://<netloc>/<path>;<params>?<query>#<fragment>
        # return all([result.scheme, result.netloc])
        netloc = result.netloc.lower()
        path = result.path.lower()
        # Regular expression pattern for YouTube video URLs
        youtube_pattern = re.compile(
            r'(https?://)?(www\.)?'
            r'(youtube\.com/watch\?v=|youtu\.be/)'
            r'(?P<video_id>[A-Za-z0-9_-]{11})'
        )
        # Check if the URL matches the pattern
        match = youtube_pattern.match(url)
        if match:
            return "youtube"
        
        return "other"
    except ValueError:
        return "invalid"
 