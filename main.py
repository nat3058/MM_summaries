from youtube_transcript_api import YouTubeTranscriptApi

from youtube_transcript_api.formatters import JSONFormatter
from youtube_transcript_api.formatters import TextFormatter
from bs4 import BeautifulSoup
from pytube import Playlist
from meta_ai_api import MetaAI
import datetime
import threading
import json
import sys

from textwrap import wrap

def append_html_contents(source_file, destination_file, target_element_id):
    """
    Appends the contents of one HTML file to a specific element in another.

    Args:
        source_file (str): Path to the source HTML file.
        destination_file (str): Path to the destination HTML file.
        target_element_id (str): ID of the element where the source contents will be appended.
    """

    try:
        # Parse the source file
        with open(source_file, 'r') as file:
            source_soup = BeautifulSoup(file, 'html.parser')

        # Parse the destination file
        with open(destination_file, 'r') as file:
            destination_soup = BeautifulSoup(file, 'html.parser')

        # Find the target element in the destination file
        target_element = destination_soup.find(id=target_element_id)

        if target_element:
            # Append the source contents to the target element
            target_element.append(source_soup)

            # Write the updated destination contents to the file
            with open(source_file, 'w') as file:
                file.write(str(destination_soup))

            print("HTML contents appended successfully.")
        else:
            print(f"Target element with ID '{target_element_id}' not found.")

    except FileNotFoundError:
        print("File not found. Please check the file paths.")
    except Exception as e:
        print(f"An error occurred: {e}")

def append_html_contents_string(source_string, destination_file, target_element_id):
    """
    Appends the contents of an HTML string to a specific element in another HTML file.

    Args:
        source_string (str): The HTML string to be appended.
        destination_file (str): Path to the destination HTML file.
        target_element_id (str): ID of the element where the source contents will be appended.
    """

    try:
        # Parse the source string
        source_soup = BeautifulSoup(source_string, 'html.parser')

        # Parse the destination file
        with open(destination_file, 'r') as file:
            destination_soup = BeautifulSoup(file, 'html.parser')

        # Find the target element in the destination file
        target_element = destination_soup.find(id=target_element_id)

        if target_element:
            # Append the source contents to the target element
            target_element.append(source_soup)

            # Write the updated destination contents to the file
            with open(destination_file, 'w') as file:
                file.write(str(destination_soup))

            print("HTML contents appended successfully.")
        else:
            print(f"Target element with ID '{target_element_id}' not found.")

    except FileNotFoundError:
        print("File not found. Please check the file paths.")
    except Exception as e:
        print(f"An error occurred: {e}")





def get_video_ids(playlist_url):
    """
    Retrieves video IDs from a public YouTube playlist.

    Args:
        playlist_url (str): URL of the YouTube playlist.
    """

    # Create a Playlist object
    p = Playlist(playlist_url)

    # Get video URLs
    video_urls = p.video_urls

    # Extract video IDs from URLs
    video_ids = [url.split('watch?v=')[-1] for url in video_urls]

    # Print video IDs
    return(video_ids)




print("Starting program...")
# Get today's date
today = datetime.date.today()

# Format the date (YYYY-MM-DD)
formatted_date = today.strftime("%Y-%m-%d")

# Print the formatted date
print(formatted_date)

def get_previous_show_dates(n):
    # Don't include weekends
    today = datetime.date.today()
    previous_days = []
    i = 0
    while len(previous_days) < n:
        previous_day = today - datetime.timedelta(days=i+1)
        if previous_day.weekday() < 5:  # 5 = Saturday, 6 = Sunday
            formatted_date = previous_day.strftime("%Y-%m-%d")
            previous_days.append(formatted_date)
        i += 1
    return previous_days
RANGE_OF_DAYS_TO_SUMM = 20
previous_days_list = get_previous_show_dates(RANGE_OF_DAYS_TO_SUMM)
print(previous_days_list)


# uniq_summary_file_name = "MM_summary_"
# uniq_summary_file_name += formatted_date
# uniq_summary_file_name += ".html"
# source_file_path = uniq_summary_file_name # summary file
# destination_file_path = 'boilerplate.html' # boiler plate fil
# target_element_id = 'append_here'
# append_html_contents(source_file_path, destination_file_path, target_element_id)


video_ids = get_video_ids("https://www.youtube.com/playlist?list=PLVbP054jv0KoZTJ1dUe3igU7K-wUcQsCI")
print(video_ids)
print(" ** Video IDs ready")


def get_summary(formatted_date, v_id):
    formatted_date = formatted_date
    video_id = v_id
    print("current video id: ")
    print(video_id)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)


    print("\n * Formatting into chunks...")
    # lets do text formatting
    formatter = TextFormatter()
    txt_formatted = formatter.format_transcript(transcript)
    #print(txt_formatted)
    with open('transcript_date.txt', 'w') as txt_file:
        txt_file.write(txt_formatted)


    # preprocess transcript
    # Specify the text after which you want to remove the rest
    stop_text = "all opinions expressed by Jim Kramer on this podcast are solely"

    # Find the index of the stop text
    stop_index = txt_formatted.find(stop_text)

    # If the stop text is found, remove everything after it
    if stop_index != -1:
        txt_formatted = txt_formatted[:stop_index]

    chnks = wrap(txt_formatted, width=14000)

    # for c in chnks:
    #     print("\n\n\nNEW CHUNK\n");
    #     print(c)


    print("\n * Asking meta ai...")

    meta = MetaAI()
    set_stage_req = "I will give u a transcript of an podcast in multiple installments since it is too long. Please be sure to read each and understand it. Once, i have given u the entire transcript, then I will ask for a summary"

    # #response = ai.prompt(message="Whats the weather in San Francisco today? And what is the date?")
    # #print(response)
    #print(meta.prompt("what is 2 + 2?"))
    # print(meta.prompt("what was my previous question?"))
    #print(meta.prompt(set_stage_req))

    print(meta.prompt(set_stage_req))

    install_num = 1;
    for c in chnks:
        installment_req = 'Here is installment number '
        installment_req += str(install_num)
        installment_req += ': \n'
        installment_req += c
        # print(installment_req)
        print(meta.prompt(installment_req))
        install_num = install_num + 1

    ending_summary_req = '''

    nice! I've given u all installments of the transcript. 
    Please give me a very detailed summary with good headings of the transcript. 
    Thanks!

    '''
    p9='''I am an data-driven knowledge-hungry stock market investor but did not watch Jim Cramer's show yet. 
Please provide me a specific recap of the entire show in the same order in which the info/segment was presented.
Make sure that information is grouped by segments directly from the show, 
and each segment has a clear header with detailed concise sub bullet points. 
Don't add information that isn't really helpful or useful to know as a investor.'''

    
    # Make the summary into a HTML format that looks pretty see on the web.
    # The summary should be detailed enough that an regular investor has a good understanding of the key details of the podcast 
    # (even if they might not have listened to the podcast).

    #final_summary = meta.prompt(ending_summary_req) 
    final_summary = meta.prompt(p9)

    # print(final_summary)
    # print(final_summary["message"])

    final_summary = meta.prompt("this is great but can u make this into a HTML format that looks pretty see on the web. Don't add any CSS please")
    print("* HTML formatting done ")
    final_summary = meta.prompt("please make the summary more detailed and includes relevant info for retail investors")
    print("* more details done ")
    # final_summary = meta.prompt("can you make sure that the summary is more detailed such that each section has at least 5 bullet points")
    # print("* 5 bullets each done ")

    uniq_summary_file_name = "./web/MM_summary_"
    uniq_summary_file_name += formatted_date
    uniq_summary_file_name += ".html"

    with open('transcript_summary.txt', 'w') as txt_file:
        txt_file.write(final_summary["message"])
    # with open('transcript_summary_html.html', 'w') as txt_file:
    #         txt_file.write(final_summary["message"])
    with open(uniq_summary_file_name, 'w') as txt_file:
            txt_file.write(final_summary["message"])

    source_file_path = uniq_summary_file_name # summary file
    destination_file_path = 'boilerplate.html' # boiler plate fil
    target_element_id = 'append_here'

    # add the src file to the destination file and then overwrite the src file with this newly appeneded file
    append_html_contents(source_file_path, destination_file_path, target_element_id)

    destination_file_path = uniq_summary_file_name
    src_str = "Show Date: "
    src_str += today.strftime("%A, %B %d, %Y") # show date in easy to read format
    target_element_id = "show_date_append_here"
    append_html_contents_string(src_str, destination_file_path, target_element_id)

    destination_file_path = uniq_summary_file_name
    src_str = "<a href="
    src_str += "https://www.youtube.com/watch?v="
    src_str += video_id
    src_str += ">Listen to the audio here</a>"
    target_element_id ="append_audio_link_here"
    append_html_contents_string(src_str, destination_file_path, target_element_id)

    done_msg = "*** DONE CREATING SUMMARY FOR "
    done_msg += formatted_date
    print(done_msg)
    print("** exiting thread: {}".format(threading.current_thread().name))




t_list = []
for i in range(RANGE_OF_DAYS_TO_SUMM-1):
    thread_name = "t"
    thread_name += str(i)
    # formatted_date = previous_days_list[i]
    # video_id = video_ids[i+1]

    fd = previous_days_list[i]
    v_id = video_ids[i]

    def excepthook(args):
        print(f"Exception in thread: {args.exc_value}")
        sys.exit(1)  # Exit the program

    threading.excepthook = excepthook
    print(" ** starting thread " + thread_name)
    t = threading.Thread(target=get_summary, name=thread_name,args=(fd,v_id) )
    t_list.append(t)
    t.start() 
for t in t_list:
    t.join()
print("All threads finished")



# # video_id = "EqTgkQfkZyY"
# for i in range(9):
#     formatted_date = previous_days_list[i]
#     video_id = video_ids[i+1]
#     print("current video id: ")
#     print(video_id)
#     transcript = YouTubeTranscriptApi.get_transcript(video_id)

#     #print(transcript);

#     # formatter = JSONFormatter()

#     # # .format_transcript(transcript) turns the transcript into a JSON string.
#     # # json_formatted = formatter.format_transcript(transcript)
#     # # print(json_formatted)



#     print("\n * Formatting into chunks...")
#     # lets do text formatting
#     formatter = TextFormatter()
#     txt_formatted = formatter.format_transcript(transcript)
#     #print(txt_formatted)
#     with open('transcript_date.txt', 'w') as txt_file:
#         txt_file.write(txt_formatted)


#     chnks = wrap(txt_formatted, width=14000)

#     # for c in chnks:
#     #     print("\n\n\nNEW CHUNK\n");
#     #     print(c)


#     print("\n * Asking meta ai...")

#     meta = MetaAI()
#     set_stage_req = "I will give u a transcript of an podcast in multiple installments since it is too long. Please be sure to read each and understand it. Once, i have given u the entire transcript, then I will ask for a summary"

#     # #response = ai.prompt(message="Whats the weather in San Francisco today? And what is the date?")
#     # #print(response)
#     #print(meta.prompt("what is 2 + 2?"))
#     # print(meta.prompt("what was my previous question?"))
#     #print(meta.prompt(set_stage_req))

#     print(meta.prompt(set_stage_req))

#     install_num = 1;
#     for c in chnks:
#         installment_req = 'Here is installment number '
#         installment_req += str(install_num)
#         installment_req += ': \n'
#         installment_req += c
#         print(installment_req)
#         print(meta.prompt(installment_req))
#         install_num = install_num + 1



#     #ending_summary_req = "nice! I've given u all installments of the transcript. Please provide a very extremely detailed summary of the transcript in HTML format. For that summary, please make headings in bold (using <b>) and info under the headings should be in bullet points (HTML style like <ul> and <li> ).  Thanks"
#     # ending_summary_req = '''

#     # nice! I've given u all installments of the transcript. 
#     # Please give me a very detailed summary with good headings of the transcript. 
#     # Below, i have given a sample summary to use as a guide. Thanks!


#     # Detailed Summary of Jim Cramer's Mad Money Podcast
#     # Introduction and Federal Reserve Discussion
#     # * Jim Cramer discusses the Federal Reserve's 50-basis-point rate cut and its impact on the stock market.
#     # * He highlights stocks that tend to perform well during an easing cycle, such as Apple, Target, and Textron.
#     # * Cramer notes that the rate cut is a positive sign for the economy and the stock market.
#     # * He mentions that FedEx reported disappointing earnings and cut its full-year forecast.
#     # * Cramer emphasizes the importance of understanding the impact of interest rates on individual stocks and sectors.
#     # Interview with George Kurtz, CEO of CrowdStrike
#     # * CrowdStrike's collaboration with Microsoft demonstrates industry cooperation in cybersecurity.
#     # * Kurtz discusses the importance of consolidation on the CrowdStrike platform.
#     # * He emphasizes the need for best-of-breed solutions and partnerships to provide comprehensive security.
#     # * CrowdStrike focuses on endpoint protection, data ingestion, and next-generation SIM and cloud security.
#     # * Kurtz notes that CrowdStrike's Falcon platform allows for solving multiple use cases beyond endpoint protection.
#     # * He highlights the importance of resilience by design to protect against increasingly complex threats.
#     # Darden Restaurants Earnings Report
#     # * Darden Restaurants (DRI) reports disappointing earnings, but the stock surges 8% due to:
#     #     * Robust forecast for 2025 fiscal year.
#     #     * Partnership with Uber Eats for delivery services.
#     #     * Improved commodity inflation outlook.
#     # * Cramer notes that the consumer spending concerns may be overblown.
#     # * Darden's same-store sales declined, but the company maintained its forecast.
#     # * The partnership with Uber Eats is expected to drive growth and increase customer convenience.
#     # Lightning Round
#     # * Cramer answers listener questions about stocks, including:
#     #     * Banking (BKNG): buy due to strong fundamentals and growth prospects.
#     #     * Gentex (GNTX): avoid due to auto industry concerns and declining sales.
#     #     * Cadre Holdings: positive outlook due to strong management and growth potential.
#     #     * Celsius Holdings (CELH): concerns about lack of transparency and declining sales.
#     #     * BKNG's diversified business model and strong financials make it a compelling buy.
#     # Apple iPhone 16 Pre-Sales Discussion
#     # * Initial negative reports on iPhone 16 pre-sales were debunked by T-Mobile CEO Mike Sievert.
#     # * Sievert cites strong demand and a potentially elongated launch cycle due to upcoming AI features.
#     # * Cramer advises investors to trust reputable sources and avoid anecdotal evidence.
#     # * Apple's iPhone 16 pre-sales were strong, despite initial reports suggesting weakness.
#     # * Sievert notes that T-Mobile's iPhone sales were incredible, with a successful back-to-school season.
#     # * Cramer emphasizes the importance of patience and long-term thinking when investing in quality stocks like Apple.
#     # Conclusion and Investment Advice
#     # * Cramer emphasizes the importance of trusting reliable sources and avoiding speculative reports.
#     # * He advises investors to hold onto quality stocks like Apple rather than trading based on unverified rumors.
#     # * Cramer concludes that consumer spending concerns may be overblown, and the economy remains solid.
#     # * Investors should focus on long-term growth prospects and fundamental analysis rather than short-term market fluctuations.
#     # * Cramer recommends owning quality stocks with strong fundamentals, rather than trying to time the market.
#     # * He encourages investors to be cautious of speculative reports and to verify information through reputable sources.


#     # '''

#     ending_summary_req = '''

#     nice! I've given u all installments of the transcript. 
#     Please give me a very detailed summary with good headings of the transcript. 
#     Thanks!


#     '''
#     # Make the summary into a HTML format that looks pretty see on the web.
#     # The summary should be detailed enough that an regular investor has a good understanding of the key details of the podcast 
#     # (even if they might not have listened to the podcast).

#     final_summary = meta.prompt(ending_summary_req) 
#     print(final_summary)
#     print(final_summary["message"])

#     final_summary = meta.prompt("this is great but can u make this into a HTML format that looks pretty see on the web. Don't add any CSS please")
#     print("* HTML formatting done ")
#     final_summary = meta.prompt("can you make sure that the summary is more detailed such that each section has at least 5 bullet points")
#     print("* 5 bullets each done ")

#     uniq_summary_file_name = "./web/MM_summary_"
#     uniq_summary_file_name += formatted_date
#     uniq_summary_file_name += ".html"

#     with open('transcript_summary.txt', 'w') as txt_file:
#         txt_file.write(final_summary["message"])
#     # with open('transcript_summary_html.html', 'w') as txt_file:
#     #         txt_file.write(final_summary["message"])
#     with open(uniq_summary_file_name, 'w') as txt_file:
#             txt_file.write(final_summary["message"])

#     source_file_path = uniq_summary_file_name # summary file
#     destination_file_path = 'boilerplate.html' # boiler plate fil
#     target_element_id = 'append_here'

#     # add the src file to the destination file and then overwrite the src file with this newly appeneded file
#     append_html_contents(source_file_path, destination_file_path, target_element_id)

#     destination_file_path = uniq_summary_file_name
#     src_str = "Show Date: "
#     src_str += today.strftime("%A, %B %d, %Y") # show date in easy to read format
#     target_element_id = "show_date_append_here"
#     append_html_contents_string(src_str, destination_file_path, target_element_id)

#     destination_file_path = uniq_summary_file_name
#     src_str = "<a href="
#     src_str += "https://www.youtube.com/watch?v="
#     src_str += video_id
#     src_str += ">Listen to the audio here</a>"
#     target_element_id ="append_audio_link_here"
#     append_html_contents_string(src_str, destination_file_path, target_element_id)

#     done_msg = "*** DONE CREATING SUMMARY FOR "
#     done_msg += formatted_date
#     print(done_msg)


# FOR DEUGGING AND TESTING
    
# while True:
#     user_input = input("Please enter something (type 'q' to exit): ")
#     if user_input.lower() == "q":
#         break
#     # Perform your desired action with the user_input
#     print(f"You entered: {user_input}")
#     meta_op = meta.prompt(user_input) 
#     print(meta_op)
#     print(meta_op["message"])
#     with open('transcript_summary.txt', 'w') as txt_file:
#         txt_file.write(meta_op["message"])
#     with open('transcript_summary_html.html', 'w') as txt_file:
#         txt_file.write(meta_op["message"])


