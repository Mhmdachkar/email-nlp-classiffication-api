#!/usr/bin/env python3
"""
🎯 ENHANCED DATASET GENERATOR
=============================
Generates high-quality, long-format email datasets for BART fine-tuning.
Includes realistic email structures, headers, signatures, and content.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import re

class EnhancedEmailDatasetGenerator:
    """Generates realistic, long-format email datasets for training."""
    
    def __init__(self):
        self.companies = [
            "Microsoft", "Google", "Amazon", "Apple", "Meta", "Tesla", "Netflix", 
            "Salesforce", "Adobe", "Oracle", "IBM", "Cisco", "Intel", "Nvidia"
        ]
        
        self.departments = [
            "Engineering", "Marketing", "Sales", "HR", "Finance", "Operations", 
            "Legal", "IT", "Product", "Design", "Security", "Customer Success"
        ]
        
        self.names = [
            "Alex Johnson", "Sarah Chen", "Michael Rodriguez", "Emily Davis", 
            "David Thompson", "Lisa Wang", "Robert Brown", "Jennifer Miller",
            "Christopher Lee", "Amanda Wilson", "James Garcia", "Maria Lopez",
            "William Taylor", "Jessica Anderson", "Thomas Martinez", "Ashley Jones"
        ]
        
        self.email_domains = [
            "company.com", "corp.com", "tech.com", "enterprise.com", "business.com",
            "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"
        ]
        
    def generate_email_header(self, category: str) -> Dict[str, str]:
        """Generate realistic email headers."""
        sender_name = random.choice(self.names)
        sender_domain = random.choice(self.email_domains)
        
        if category == "Work":
            sender_domain = random.choice(self.email_domains[:5])  # Corporate domains
        elif category == "Personal":
            sender_domain = random.choice(self.email_domains[5:])  # Personal domains
            
        sender_email = f"{sender_name.lower().replace(' ', '.')}.{random.randint(1, 999)}@{sender_domain}"
        
        # Generate timestamp (last 30 days)
        base_date = datetime.now()
        random_days = random.randint(0, 30)
        email_date = base_date - timedelta(days=random_days)
        
        return {
            "from": f"{sender_name} <{sender_email}>",
            "date": email_date.strftime("%a, %d %b %Y %H:%M:%S %z"),
            "sender_name": sender_name,
            "sender_email": sender_email
        }
    
    def generate_email_signature(self, category: str, sender_name: str) -> str:
        """Generate realistic email signatures."""
        if category == "Work":
            company = random.choice(self.companies)
            department = random.choice(self.departments)
            title = random.choice([
                "Senior Manager", "Director", "Vice President", "Lead Engineer",
                "Product Manager", "Business Analyst", "Project Manager", "Specialist"
            ])
            
            signatures = [
                f"""
Best regards,
{sender_name}
{title}, {department}
{company}
Phone: +1 (555) {random.randint(100, 999)}-{random.randint(1000, 9999)}
Email: {sender_name.lower().replace(' ', '.')}.{random.randint(1, 999)}@{company.lower()}.com

This email and any attachments are confidential and may be privileged.
""",
                f"""
Thank you,

{sender_name}
{title} | {department} Department
{company}
Direct: +1 (555) {random.randint(100, 999)}-{random.randint(1000, 9999)}
Mobile: +1 (555) {random.randint(100, 999)}-{random.randint(1000, 9999)}

{company} - Innovating for the future
""",
                f"""
Kind regards,
{sender_name}

{title}, {department}
{company}
Office: +1 (555) {random.randint(100, 999)}-{random.randint(1000, 9999)}

Please consider the environment before printing this email.
"""
            ]
        else:
            signatures = [
                f"\nBest,\n{sender_name}",
                f"\nThanks!\n{sender_name}",
                f"\nCheers,\n{sender_name}",
                f"\nWarm regards,\n{sender_name}",
                f"\n- {sender_name}",
                f"\nTalk soon,\n{sender_name}"
            ]
        
        return random.choice(signatures)
    
    def generate_spam_emails(self, count: int) -> List[Dict]:
        """Generate realistic spam emails with long format."""
        spam_emails = []
        
        spam_templates = [
            {
                "subject": "🎉 CONGRATULATIONS! You've Won ${amount} - Claim Now!",
                "content": """Dear Lucky Winner,

CONGRATULATIONS! This is to officially notify you that you have been selected as one of our GRAND PRIZE WINNERS in the {lottery_name} International Email Lottery Program.

YOUR WINNING DETAILS:
- Prize Amount: ${amount}
- Winning Number: {winning_number}
- Reference Number: {ref_number}
- Batch Number: {batch_number}

Your email address was randomly selected from over 2.5 million email addresses worldwide. This lottery is organized by {organization} in partnership with leading technology companies.

TO CLAIM YOUR PRIZE:
1. Reply to this email with your full name and phone number
2. Provide a copy of your government-issued ID
3. Pay the processing fee of ${processing_fee} via Western Union
4. Your prize will be deposited within 48 hours

IMPORTANT: This offer expires in 72 hours. Do not share this information with anyone to avoid disqualification.

Contact our claims department immediately:
Email: claims.{lottery_name}@{domain}
Phone: +44 {phone_number}

We look forward to making you our next millionaire!

Best regards,
Dr. {manager_name}
Prize Coordinator
{organization}

*This email is legitimate and verified by international lottery commission."""
            },
            {
                "subject": "🔒 URGENT: Your {service} Account Security Alert",
                "content": """Dear {service} User,

We have detected suspicious activity on your {service} account. For your security, we have temporarily limited access to your account.

SECURITY ALERT DETAILS:
- Date: {date}
- Time: {time}
- Location: {location}
- IP Address: {ip_address}
- Device: Unknown Device

IMMEDIATE ACTION REQUIRED:
We need you to verify your account information to restore full access. Failure to verify within 24 hours will result in permanent account suspension.

STEPS TO VERIFY YOUR ACCOUNT:
1. Click the secure verification link below
2. Enter your account credentials
3. Confirm your identity with security questions
4. Update your payment information if required

🔗 VERIFY YOUR ACCOUNT NOW: https://secure-{service}-verification.{domain}/verify?token={token}

If you did not authorize this activity, your account may have been compromised. Please verify immediately to secure your account.

For additional security:
- Never share your password with anyone
- Use strong, unique passwords
- Enable two-factor authentication
- Keep your contact information updated

If you have any questions, contact our 24/7 security team:
Email: security@{service}.com
Phone: 1-800-{phone}

Thank you for using {service}.

{service} Security Team
This is an automated security notification."""
            },
            {
                "subject": "💊 Premium Health Products - 80% OFF Limited Time",
                "content": """Hello Health-Conscious Friend,

Are you tired of feeling tired? Struggling with low energy, poor sleep, or unwanted weight? 

INTRODUCING THE REVOLUTIONARY {product_name}™ SYSTEM

Our scientifically-formulated supplements have helped over 100,000 people worldwide achieve:
✅ Increased Energy Levels (up to 300% boost)
✅ Better Sleep Quality (fall asleep in under 10 minutes)
✅ Rapid Weight Loss (lose 20+ pounds in 30 days)
✅ Enhanced Mental Clarity (improve focus by 200%)
✅ Stronger Immune System (99% fewer sick days)

EXCLUSIVE LIMITED-TIME OFFER:
🎯 80% OFF Regular Price
🎯 FREE Shipping Worldwide
🎯 60-Day Money-Back Guarantee
🎯 FREE Bonus: {bonus_product} (worth $97)

CUSTOMER TESTIMONIALS:
"I lost 25 pounds in just 3 weeks! This product changed my life!" - Sarah M., California
"My energy levels are through the roof. I feel 20 years younger!" - Mike R., Texas
"Best investment I've ever made for my health." - Jennifer L., New York

SCIENTIFIC BACKING:
Our formula contains patent-pending ingredients clinically proven in over 15 studies:
- {ingredient1}: Increases metabolism by 400%
- {ingredient2}: Blocks fat absorption by 75%
- {ingredient3}: Reduces cortisol levels by 60%

⏰ HURRY! Only {remaining} bottles left in stock!

ORDER NOW and join thousands of satisfied customers who have transformed their lives.

🛒 CLICK HERE TO ORDER: www.{product_domain}.com/order

Questions? Call our 24/7 hotline: 1-800-{phone_number}

To your health and success,
Dr. {doctor_name}
Chief Medical Officer
{company_name}

P.S. This 80% discount expires at midnight tonight. Don't miss out on this life-changing opportunity!

*Results may vary. Not evaluated by FDA."""
            }
        ]
        
        for i in range(count):
            template = random.choice(spam_templates)
            header = self.generate_email_header("Spam")
            
            # Fill template variables
            content = template["content"]
            variables = {
                "amount": f"{random.randint(50, 500)},000",
                "lottery_name": random.choice(["Euro", "Global", "International", "Mega", "Super"]),
                "winning_number": f"LT{random.randint(10000, 99999)}",
                "ref_number": f"REF/{random.randint(1000, 9999)}/WIN",
                "batch_number": f"BTH{random.randint(100, 999)}",
                "organization": random.choice(["Microsoft Corporation", "Google Foundation", "Apple Charitable Trust"]),
                "processing_fee": random.randint(50, 500),
                "domain": random.choice(["lottery-claims.org", "prize-center.net", "winner-support.com"]),
                "phone_number": f"{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "manager_name": random.choice(self.names),
                "service": random.choice(["Netflix", "Amazon", "Microsoft", "Google", "Apple"]),
                "date": datetime.now().strftime("%B %d, %Y"),
                "time": f"{random.randint(1, 12)}:{random.randint(10, 59)} {'AM' if random.random() > 0.5 else 'PM'}",
                "location": random.choice(["Unknown Location", "Foreign Country", "Suspicious Location"]),
                "ip_address": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
                "token": f"tk{random.randint(100000, 999999)}",
                "phone": f"{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "product_name": random.choice(["VitalMax", "HealthBoost", "EnergyPro", "WellnessPrime"]),
                "bonus_product": random.choice(["Immunity Booster", "Sleep Aid", "Detox Cleanse"]),
                "ingredient1": random.choice(["Garcinia Cambogia", "Green Coffee Extract", "Forskolin"]),
                "ingredient2": random.choice(["Chitosan", "White Kidney Bean", "Glucomannan"]),
                "ingredient3": random.choice(["Ashwagandha", "Rhodiola Rosea", "L-Theanine"]),
                "remaining": random.randint(5, 50),
                "product_domain": random.choice(["health-miracle", "vita-supplements", "wellness-store"]),
                "phone_number": f"{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "doctor_name": random.choice(self.names),
                "company_name": random.choice(["HealthTech Solutions", "Wellness Innovations", "VitalLife Corp"])
            }
            
            for key, value in variables.items():
                content = content.replace(f"{{{key}}}", str(value))
            
            full_email = f"From: {header['from']}\nDate: {header['date']}\nSubject: {template['subject']}\n\n{content}"
            
            spam_emails.append({
                "email": full_email,
                "category": "Spam",
                "content_length": len(full_email)
            })
        
        return spam_emails
    
    def generate_work_emails(self, count: int) -> List[Dict]:
        """Generate realistic work emails with long format."""
        work_emails = []
        
        work_templates = [
            {
                "subject": "Q4 Budget Review Meeting - Action Items and Next Steps",
                "content": """Hi Team,

Thank you all for attending today's Q4 budget review meeting. I wanted to follow up with the key discussion points and action items we covered.

MEETING SUMMARY:
Date: {date}
Attendees: {attendees}
Duration: 2 hours

KEY DISCUSSION POINTS:

1. BUDGET ALLOCATION REVIEW
   - Marketing budget increased by 15% to support new product launch
   - Engineering resources reallocated to prioritize security initiatives
   - Travel budget reduced by 30% due to continued remote work policies

2. PROJECT PRIORITIZATION
   - Project Alpha: On track for Q1 2024 delivery
   - Project Beta: Delayed 2 weeks due to resource constraints
   - Project Gamma: Budget approved for additional contractor support

3. DEPARTMENTAL UPDATES
   {department_updates}

ACTION ITEMS:
□ {action1} (Owner: {owner1}, Due: {due1})
□ {action2} (Owner: {owner2}, Due: {due2})
□ {action3} (Owner: {owner3}, Due: {due3})
□ Submit revised budget proposals by {budget_due}
□ Schedule follow-up meetings with stakeholders

NEXT STEPS:
- Individual department heads will receive detailed budget breakdowns by COB Friday
- All action items to be reviewed in next week's leadership meeting
- Final budget approval expected by {approval_date}

BUDGET CONCERNS:
Please note that all expenses over $5,000 now require VP approval. This includes:
- Software licenses and subscriptions
- External contractor agreements
- Conference and training expenses
- Equipment purchases

If you have any questions about your department's allocation or need clarification on any of the action items, please don't hesitate to reach out. I'm available for one-on-one discussions this week.

Let's make Q4 a successful quarter for the entire organization!
"""
            },
            {
                "subject": "Client Presentation Preparation - Urgent Deadline Friday",
                "content": """Dear Team,

I hope this email finds you well. I'm writing to brief you on the upcoming client presentation for {client_name}, scheduled for {presentation_date}.

PROJECT BACKGROUND:
{client_name} is a {industry} company with annual revenue of ${revenue}M. They're looking to modernize their technology infrastructure and have shortlisted us among three vendors for this ${project_value}M contract.

PRESENTATION DETAILS:
Date: {presentation_date}
Time: {presentation_time}
Location: {location}
Duration: 90 minutes + 30 minutes Q&A
Audience: {audience}

AGENDA OUTLINE:
1. Company Introduction (10 minutes) - Sarah
2. Technical Solution Overview (25 minutes) - Engineering Team
3. Implementation Timeline (15 minutes) - Project Management
4. Cost Analysis and ROI (20 minutes) - Finance Team
5. Case Studies and References (15 minutes) - Sales Team
6. Q&A Session (30 minutes) - All

PREPARATION REQUIREMENTS:

ENGINEERING TEAM:
- Prepare technical architecture diagrams
- Demo environment setup and testing
- Performance benchmarks comparison
- Security compliance documentation

SALES TEAM:
- Competitor analysis report
- Pricing strategy justification
- Reference customer contacts
- Contract terms preparation

PROJECT MANAGEMENT:
- Detailed project timeline with milestones
- Resource allocation plan
- Risk assessment and mitigation strategies
- Quality assurance procedures

CRITICAL SUCCESS FACTORS:
1. Demonstrate clear understanding of their business challenges
2. Show how our solution addresses their specific pain points
3. Provide concrete examples of successful implementations
4. Present realistic timelines and budgets
5. Establish trust and credibility with technical expertise

LOGISTICS:
- All materials must be finalized by Thursday 5 PM
- Dress code: Business formal
- Backup presentation equipment will be available
- Catering arranged for lunch meeting
- Parking instructions attached

CLIENT BACKGROUND RESEARCH:
I've attached detailed information about {client_name}, including:
- Annual reports for the last 3 years
- Recent press releases and news articles
- LinkedIn profiles of key decision makers
- Their current technology stack analysis
- Competitor solutions they're evaluating

This is a significant opportunity for our company. Success here could lead to additional projects worth ${additional_value}M over the next two years.

Please confirm your availability and let me know if you need any additional resources or support for preparation.
"""
            }
        ]
        
        for i in range(count):
            template = random.choice(work_templates)
            header = self.generate_email_header("Work")
            signature = self.generate_email_signature("Work", header["sender_name"])
            
            # Fill template variables
            content = template["content"]
            variables = {
                "date": datetime.now().strftime("%B %d, %Y"),
                "attendees": ", ".join(random.sample(self.names, 5)),
                "department_updates": self._generate_department_updates(),
                "action1": random.choice([
                    "Finalize vendor contracts for new software licenses",
                    "Complete security audit for all systems",
                    "Submit quarterly performance reports"
                ]),
                "owner1": random.choice(self.names),
                "due1": (datetime.now() + timedelta(days=random.randint(3, 10))).strftime("%B %d"),
                "action2": random.choice([
                    "Review and approve marketing campaign budgets",
                    "Conduct team performance evaluations",
                    "Implement new project management workflows"
                ]),
                "owner2": random.choice(self.names),
                "due2": (datetime.now() + timedelta(days=random.randint(5, 15))).strftime("%B %d"),
                "action3": random.choice([
                    "Organize cross-departmental training sessions",
                    "Evaluate new office space requirements",
                    "Update employee handbook policies"
                ]),
                "owner3": random.choice(self.names),
                "due3": (datetime.now() + timedelta(days=random.randint(7, 20))).strftime("%B %d"),
                "budget_due": (datetime.now() + timedelta(days=7)).strftime("%B %d"),
                "approval_date": (datetime.now() + timedelta(days=14)).strftime("%B %d"),
                "client_name": f"{random.choice(['Global', 'Premier', 'Advanced', 'Strategic'])} {random.choice(['Solutions', 'Technologies', 'Systems', 'Enterprises'])}",
                "industry": random.choice(["healthcare", "finance", "manufacturing", "retail", "technology"]),
                "revenue": random.randint(50, 500),
                "project_value": random.randint(1, 10),
                "presentation_date": (datetime.now() + timedelta(days=random.randint(3, 14))).strftime("%A, %B %d, %Y"),
                "presentation_time": f"{random.randint(9, 15)}:00 AM",
                "location": random.choice([
                    "Client Conference Room A, 45th Floor",
                    "Our Main Conference Room",
                    "Virtual Meeting (Teams)",
                    "Downtown Office - Executive Boardroom"
                ]),
                "audience": random.choice([
                    "CEO, CTO, VP of Operations, IT Director",
                    "Board of Directors and Executive Team",
                    "Technical Committee and Decision Makers"
                ]),
                "additional_value": random.randint(5, 25)
            }
            
            for key, value in variables.items():
                content = content.replace(f"{{{key}}}", str(value))
            
            full_email = f"From: {header['from']}\nDate: {header['date']}\nSubject: {template['subject']}\n\n{content}{signature}"
            
            work_emails.append({
                "email": full_email,
                "category": "Work",
                "content_length": len(full_email)
            })
        
        return work_emails
    
    def _generate_department_updates(self) -> str:
        """Generate realistic department updates."""
        updates = []
        departments = random.sample(self.departments, 3)
        
        for dept in departments:
            update = random.choice([
                f"{dept}: Exceeded quarterly targets by 12%, new hires onboarding next week",
                f"{dept}: Project timeline adjusted due to resource reallocation, still on track",
                f"{dept}: Successfully completed system upgrade, performance improved by 25%",
                f"{dept}: Implementing new processes to increase efficiency and reduce costs"
            ])
            updates.append(f"   - {update}")
        
        return "\n".join(updates)
    
    def generate_personal_emails(self, count: int) -> List[Dict]:
        """Generate realistic personal emails with long format."""
        personal_emails = []
        
        personal_templates = [
            {
                "subject": "Can't wait to catch up this weekend!",
                "content": """Hey {friend_name}!

I hope you're doing amazing! It feels like forever since we last had a proper chat. I know we've been texting here and there, but there's nothing like a good old-fashioned conversation over coffee (or wine, depending on the time of day 😉).

So much has happened since we last hung out! Remember how I was telling you about {event1}? Well, it finally happened and it was absolutely incredible! I still can't believe it all worked out. I have so many stories to share with you.

WEEKEND PLANS:
I was thinking we could meet up this {day} at {location}. They have that amazing {food_item} you loved last time, and I heard they added some new items to their menu. We could grab a table outside if the weather's nice.

Time-wise, I'm flexible. I was thinking around {time}, but let me know what works better for you. I know you mentioned you had {commitment} in the morning, so afternoon might be better?

WHAT'S NEW WITH ME:
- {update1}
- {update2}
- {update3}

I'm also dying to hear about {friend_update}! Your last message made it sound like things are really looking up. And please tell me more about {friend_event} - I want all the details!

Oh, and I almost forgot - {mutual_friend} asked me to say hi! We ran into each other at {place} last week and spent like an hour talking about old times. Do you remember when we all went to {memory_place} and {funny_memory}? We were laughing so hard about that!

ALSO:
I found some old photos from {past_event} while cleaning out my apartment. You look absolutely ridiculous in them (in the best way possible), and I may or may not be planning to use them for blackmail purposes 😄. I'll bring them along so we can have a good laugh.

Speaking of apartments, the new place is finally starting to feel like home. I finished decorating the living room last weekend, and I think you'd really love how it turned out. You'll have to come over soon for a proper housewarming dinner!

Let me know if this weekend works for you, and if not, we'll figure out another time. I miss our conversations and your terrible jokes (yes, they're terrible, but I love them anyway).

Can't wait to see you!
"""
            },
            {
                "subject": "Family Reunion Planning Update",
                "content": """Hi Everyone,

I hope this email finds you and your families in good health and high spirits! As we discussed during our last family call, I wanted to share an update on the planning for our annual family reunion.

REUNION DETAILS:
Date: {reunion_date}
Location: {reunion_location}
Duration: {duration}
Expected Attendees: {attendee_count} family members

ACCOMMODATION UPDATE:
I've been in touch with {hotel_name}, and they've agreed to hold a block of rooms for us at a discounted rate of ${room_rate} per night. The deadline to book at this rate is {booking_deadline}, so please make your reservations soon!

For those preferring vacation rentals, I found several large houses on {rental_platform} that could accommodate multiple families. {cousin_name} and {relative_name} have already expressed interest in sharing a rental. Let me know if you'd like to join them or prefer other arrangements.

ACTIVITY PLANNING:
Based on everyone's feedback, here's what we have planned so far:

SATURDAY ACTIVITIES:
- 10:00 AM: Welcome breakfast at the pavilion
- 12:00 PM: Family photo session (professional photographer booked!)
- 2:00 PM: Outdoor games and sports for all ages
- 4:00 PM: Storytelling session with Grandpa and Grandma
- 6:00 PM: Barbecue dinner (catered by {catering_company})
- 8:00 PM: Campfire and s'mores

SUNDAY ACTIVITIES:
- 9:00 AM: Memorial service for those we've lost
- 11:00 AM: Brunch and family meeting
- 1:00 PM: Free time for smaller group activities
- 3:00 PM: Farewell gathering

SPECIAL ARRANGEMENTS:
- Wheelchair accessible facilities confirmed
- Vegetarian and dietary restriction options available
- Kids' corner with games and activities
- Photo booth with family props
- Live music during dinner (Uncle {musician_name} volunteered!)

WHAT TO BRING:
- Family recipes for the recipe exchange
- Old photos for the memory wall
- Comfortable clothes and shoes
- Sunscreen and bug spray
- Folding chairs (if you have them)
- Your appetites and good humor!

COST BREAKDOWN:
The total cost per family is estimated at ${family_cost}, which covers:
- All meals and refreshments
- Professional photography
- Activity supplies and equipment
- Facility rental fees
- Welcome gifts for everyone

Please send your payment to {payment_person} by {payment_deadline}. They accept cash, check, or Venmo (@{venmo_handle}).

RSVP STATUS:
Confirmed: {confirmed_families}
Pending: {pending_families}
Unable to attend: {unable_families}

If you haven't responded yet, please let me know by {rsvp_deadline} so we can finalize catering numbers.

I'm so excited to see everyone and create new memories together! It's been too long since we've all been in the same place. The kids have grown so much, and there are new family members to meet!

If you have any questions, suggestions, or concerns, please don't hesitate to reach out. You can call me at {phone_number} or reply to this email.

Looking forward to our wonderful family time together!

Love and hugs,
{organizer_name}

P.S. - Mom wanted me to remind everyone to bring sweaters for the evening. You know how she worries about us catching cold! 💙
"""
            }
        ]
        
        for i in range(count):
            template = random.choice(personal_templates)
            header = self.generate_email_header("Personal")
            signature = self.generate_email_signature("Personal", header["sender_name"])
            
            # Fill template variables
            content = template["content"]
            variables = {
                "friend_name": random.choice(self.names),
                "event1": random.choice([
                    "that job interview I was nervous about",
                    "the house hunting adventure",
                    "my sister's wedding planning",
                    "the cooking class I signed up for"
                ]),
                "day": random.choice(["Saturday", "Sunday"]),
                "location": random.choice([
                    "our favorite coffee shop downtown",
                    "that new brunch place on Main Street",
                    "the park near your place",
                    "the bookstore cafe we love"
                ]),
                "food_item": random.choice(["avocado toast", "pancakes", "scones", "sandwiches"]),
                "time": random.choice(["11:00 AM", "1:00 PM", "2:30 PM", "3:00 PM"]),
                "commitment": random.choice(["yoga class", "dentist appointment", "family brunch", "dog grooming"]),
                "update1": random.choice([
                    "Started a new fitness routine and actually sticking to it!",
                    "Finished redecorating my bedroom and love the new colors",
                    "Adopted a rescue kitten and she's absolutely adorable"
                ]),
                "update2": random.choice([
                    "Finally learned how to make sourdough bread from scratch",
                    "Completed my first 5K run without stopping",
                    "Started taking Italian lessons online"
                ]),
                "update3": random.choice([
                    "Planning a trip to visit my college roommate next month",
                    "Got promoted to team lead at work",
                    "Joined a local book club and making new friends"
                ]),
                "friend_update": random.choice([
                    "your new job",
                    "the dating situation",
                    "your art classes",
                    "the home renovation project"
                ]),
                "friend_event": random.choice([
                    "that concert you went to",
                    "your family vacation",
                    "the cooking competition you entered",
                    "your cousin's graduation"
                ]),
                "mutual_friend": random.choice(self.names),
                "place": random.choice([
                    "the grocery store",
                    "the gym",
                    "downtown",
                    "the farmer's market"
                ]),
                "memory_place": random.choice([
                    "that karaoke bar",
                    "the beach house",
                    "the amusement park",
                    "the music festival"
                ]),
                "funny_memory": random.choice([
                    "you tried to impress that guy with your 'singing'",
                    "we got completely lost trying to find the bathroom",
                    "you spilled ice cream all over yourself",
                    "we stayed up all night talking about everything"
                ]),
                "past_event": random.choice([
                    "college graduation",
                    "your birthday party",
                    "our road trip",
                    "New Year's Eve"
                ]),
                # Family reunion variables
                "reunion_date": (datetime.now() + timedelta(days=random.randint(30, 120))).strftime("%B %d-%d, %Y"),
                "reunion_location": random.choice([
                    "Grandma's hometown community center",
                    "Lake Tahoe Resort and Conference Center",
                    "Riverside Park Pavilion",
                    "Mountain View Lodge"
                ]),
                "duration": random.choice(["2 days", "3 days", "weekend"]),
                "attendee_count": random.randint(25, 50),
                "hotel_name": random.choice([
                    "Holiday Inn Express",
                    "Hampton Inn",
                    "Best Western",
                    "Marriott"
                ]),
                "room_rate": random.randint(89, 149),
                "booking_deadline": (datetime.now() + timedelta(days=random.randint(14, 30))).strftime("%B %d"),
                "rental_platform": random.choice(["Airbnb", "VRBO", "HomeAway"]),
                "cousin_name": random.choice(self.names),
                "relative_name": random.choice(self.names),
                "catering_company": random.choice([
                    "Joe's BBQ Catering",
                    "Family Style Catering",
                    "Southern Comfort Foods"
                ]),
                "musician_name": random.choice(self.names),
                "family_cost": random.randint(75, 150),
                "payment_person": random.choice(self.names),
                "payment_deadline": (datetime.now() + timedelta(days=random.randint(7, 21))).strftime("%B %d"),
                "venmo_handle": f"{random.choice(self.names).lower().replace(' ', '')}{random.randint(1, 99)}",
                "confirmed_families": ", ".join(random.sample(self.names, 3)),
                "pending_families": ", ".join(random.sample(self.names, 2)),
                "unable_families": ", ".join(random.sample(self.names, 1)),
                "rsvp_deadline": (datetime.now() + timedelta(days=random.randint(5, 14))).strftime("%B %d"),
                "phone_number": f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "organizer_name": random.choice(self.names)
            }
            
            for key, value in variables.items():
                content = content.replace(f"{{{key}}}", str(value))
            
            full_email = f"From: {header['from']}\nDate: {header['date']}\nSubject: {template['subject']}\n\n{content}{signature}"
            
            personal_emails.append({
                "email": full_email,
                "category": "Personal",
                "content_length": len(full_email)
            })
        
        return personal_emails
    
    def generate_urgent_emails(self, count: int) -> List[Dict]:
        """Generate realistic urgent emails with long format."""
        urgent_emails = []
        
        urgent_templates = [
            {
                "subject": "🚨 CRITICAL: Production System Outage - Immediate Action Required",
                "content": """URGENT - ALL HANDS ON DECK

Team,

We are currently experiencing a CRITICAL production system outage that began at {outage_start}. This is affecting all customer-facing services and requires immediate attention from all available technical staff.

INCIDENT DETAILS:
Incident ID: {incident_id}
Severity: CRITICAL (P0)
Start Time: {outage_start}
Affected Systems: {affected_systems}
Estimated Users Impacted: {impacted_users}
Current Status: {current_status}

IMMEDIATE SYMPTOMS:
- {symptom1}
- {symptom2}
- {symptom3}
- {symptom4}

INITIAL INVESTIGATION FINDINGS:
Our preliminary analysis indicates {initial_findings}. The root cause appears to be related to {suspected_cause}, which was likely triggered by {trigger_event}.

ERROR LOGS:
{error_timestamp}: {error_message1}
{error_timestamp2}: {error_message2}
{error_timestamp3}: {error_message3}

IMMEDIATE ACTION PLAN:
1. {action1} (Owner: {owner1}, ETA: {eta1})
2. {action2} (Owner: {owner2}, ETA: {eta2})
3. {action3} (Owner: {owner3}, ETA: {eta3})
4. {action4} (Owner: {owner4}, ETA: {eta4})

INCIDENT RESPONSE TEAM ACTIVATION:
□ Incident Commander: {ic_name} (Mobile: {ic_phone})
□ Technical Lead: {tech_lead} (Mobile: {tech_phone})
□ Communications Lead: {comm_lead} (Mobile: {comm_phone})
□ Customer Support Lead: {support_lead} (Mobile: {support_phone})

COMMUNICATION STRATEGY:
- Internal updates every 15 minutes via Slack #incident-response
- Customer status page updated every 30 minutes
- Executive briefing scheduled for {exec_briefing_time}
- Post-incident review to be scheduled once resolved

CUSTOMER IMPACT:
We estimate this outage is affecting approximately {customer_impact} of our active users. Customer Support is reporting {support_tickets} new tickets in the last hour, with the following common issues:
- {customer_issue1}
- {customer_issue2}
- {customer_issue3}

ESCALATION PROCEDURES:
If you cannot reach your assigned team members within 10 minutes, please escalate to:
1. Engineering Manager: {eng_manager} ({eng_manager_phone})
2. VP of Engineering: {vp_eng} ({vp_eng_phone})
3. CTO: {cto} ({cto_phone})

All team members are expected to join the incident response call immediately:
Conference Bridge: {conference_number}
Passcode: {passcode}
Slack Channel: #incident-{incident_id}

BUSINESS IMPACT:
Estimated revenue impact: ${revenue_impact}/hour
SLA breach threshold: {sla_threshold}
Customer escalations: {escalations} and rising

This is our highest priority. All non-critical work should be suspended until this incident is resolved. We need all hands working on this immediately.

Please acknowledge receipt of this email and confirm your availability to assist.

Thank you for your immediate attention to this critical matter.
"""
            },
            {
                "subject": "🔴 SECURITY BREACH DETECTED - Immediate Response Required",
                "content": """SECURITY INCIDENT ALERT - IMMEDIATE ACTION REQUIRED

Team,

Our security monitoring systems have detected a potential security breach at {detection_time}. This requires immediate attention from our security incident response team and all relevant stakeholders.

INCIDENT CLASSIFICATION: {classification}
INCIDENT ID: SEC-{incident_number}
DISCOVERY TIME: {detection_time}
REPORTING SOURCE: {detection_source}

INCIDENT SUMMARY:
We have detected {breach_type} affecting {affected_component}. The incident was first identified when {discovery_method}. Initial analysis suggests {initial_assessment}.

AFFECTED SYSTEMS:
Primary: {primary_system}
Secondary: {secondary_systems}
Potentially Compromised: {compromised_systems}

IMMEDIATE INDICATORS:
- {indicator1}
- {indicator2}
- {indicator3}
- {indicator4}

POTENTIAL DATA EXPOSURE:
Based on preliminary analysis, the following data types may have been accessed:
□ {data_type1} ({record_count1} records)
□ {data_type2} ({record_count2} records)
□ {data_type3} ({record_count3} records)
□ {data_type4} (Assessment ongoing)

CONTAINMENT ACTIONS TAKEN:
✅ {containment1} - Completed at {time1}
✅ {containment2} - Completed at {time2}
⏳ {containment3} - In progress (ETA: {time3})
⏳ {containment4} - Pending (ETA: {time4})

IMMEDIATE RESPONSE TEAM:
Incident Commander: {ic_name} ({ic_contact})
Security Lead: {sec_lead} ({sec_contact})
Legal Counsel: {legal_contact} ({legal_phone})
Communications: {comm_contact} ({comm_phone})
Compliance Officer: {compliance_contact} ({compliance_phone})

URGENT ACTION ITEMS:
1. {urgent_action1} (Owner: {urgent_owner1}, Due: ASAP)
2. {urgent_action2} (Owner: {urgent_owner2}, Due: {urgent_due2})
3. {urgent_action3} (Owner: {urgent_owner3}, Due: {urgent_due3})
4. {urgent_action4} (Owner: {urgent_owner4}, Due: {urgent_due4})

REGULATORY CONSIDERATIONS:
Given the nature of this incident, we may need to notify:
- {regulator1} within {notification_period1}
- {regulator2} within {notification_period2}
- Affected customers within {customer_notification}

COMMUNICATION PROTOCOL:
- Security team updates every 30 minutes via secure channel
- Executive briefing at {exec_time}
- External communications approval required from Legal
- Customer notification pending impact assessment

FORENSIC PRESERVATION:
Critical: DO NOT modify, restart, or shut down any potentially affected systems without approval from the Security team. We need to preserve evidence for forensic analysis.

Systems under preservation order:
- {forensic_system1}
- {forensic_system2}
- {forensic_system3}

CONFERENCE BRIDGE:
Secure line: {secure_number}
Access code: {secure_code}
Backup line: {backup_number}

All team members with security incident response training should join immediately. Others should remain on standby and await further instructions.

This is a Code Red security incident. All other work is suspended until further notice.

Please confirm receipt and availability to assist.
"""
            }
        ]
        
        for i in range(count):
            template = random.choice(urgent_templates)
            header = self.generate_email_header("Urgent")
            signature = self.generate_email_signature("Work", header["sender_name"])
            
            # Fill template variables
            content = template["content"]
            current_time = datetime.now()
            outage_time = current_time - timedelta(minutes=random.randint(15, 60))
            
            variables = {
                "outage_start": outage_time.strftime("%I:%M %p"),
                "incident_id": f"INC-{random.randint(10000, 99999)}",
                "affected_systems": random.choice([
                    "Web Application, Database Cluster, API Gateway",
                    "Payment Processing, User Authentication, Mobile App",
                    "Email Services, File Storage, CDN"
                ]),
                "impacted_users": f"{random.randint(50, 95)}%",
                "current_status": random.choice([
                    "Investigating root cause",
                    "Implementing emergency fixes",
                    "Escalating to senior engineers"
                ]),
                "symptom1": random.choice([
                    "500 errors on all web requests",
                    "Database connection timeouts",
                    "API response times > 30 seconds"
                ]),
                "symptom2": random.choice([
                    "User login failures across all platforms",
                    "Payment processing completely down",
                    "File upload/download not functioning"
                ]),
                "symptom3": random.choice([
                    "Mobile app crashes on startup",
                    "Email notifications not sending",
                    "Dashboard showing no data"
                ]),
                "symptom4": random.choice([
                    "Load balancer health checks failing",
                    "Background job queue backing up",
                    "Cache invalidation not working"
                ]),
                "initial_findings": random.choice([
                    "a cascade failure triggered by high memory usage",
                    "database corruption following the morning deployment",
                    "network connectivity issues between microservices"
                ]),
                "suspected_cause": random.choice([
                    "the recent database migration",
                    "increased traffic from the marketing campaign",
                    "memory leak in the payment service"
                ]),
                "trigger_event": random.choice([
                    "this morning's deployment at 9:00 AM",
                    "the scheduled maintenance window",
                    "traffic spike from the product launch"
                ]),
                "error_timestamp": outage_time.strftime("%H:%M:%S"),
                "error_message1": "FATAL: connection pool exhausted",
                "error_timestamp2": (outage_time + timedelta(minutes=5)).strftime("%H:%M:%S"),
                "error_message2": "ERROR: Redis cluster unreachable",
                "error_timestamp3": (outage_time + timedelta(minutes=10)).strftime("%H:%M:%S"),
                "error_message3": "CRITICAL: Application server out of memory",
                "action1": "Restart application servers in rolling fashion",
                "owner1": random.choice(self.names),
                "eta1": "15 minutes",
                "action2": "Investigate database connection pool settings",
                "owner2": random.choice(self.names),
                "eta2": "20 minutes",
                "action3": "Scale up Redis cluster resources",
                "owner3": random.choice(self.names),
                "eta3": "30 minutes",
                "action4": "Prepare rollback plan for morning deployment",
                "owner4": random.choice(self.names),
                "eta4": "45 minutes",
                "ic_name": random.choice(self.names),
                "ic_phone": f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "tech_lead": random.choice(self.names),
                "tech_phone": f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "comm_lead": random.choice(self.names),
                "comm_phone": f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "support_lead": random.choice(self.names),
                "support_phone": f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "exec_briefing_time": (current_time + timedelta(hours=1)).strftime("%I:%M %p"),
                "customer_impact": f"{random.randint(60, 90)}%",
                "support_tickets": random.randint(50, 200),
                "customer_issue1": "Cannot access account dashboard",
                "customer_issue2": "Payment processing failures",
                "customer_issue3": "Mobile app not loading",
                "eng_manager": random.choice(self.names),
                "eng_manager_phone": f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "vp_eng": random.choice(self.names),
                "vp_eng_phone": f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "cto": random.choice(self.names),
                "cto_phone": f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "conference_number": f"1-800-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "passcode": random.randint(100000, 999999),
                "revenue_impact": f"{random.randint(10, 100)},000",
                "sla_threshold": f"{random.randint(99, 100)}.{random.randint(0, 9)}%",
                "escalations": random.randint(5, 25),
                # Security breach variables
                "detection_time": (current_time - timedelta(minutes=random.randint(10, 45))).strftime("%I:%M %p"),
                "classification": random.choice(["CRITICAL", "HIGH", "MEDIUM"]),
                "incident_number": random.randint(2024001, 2024999),
                "detection_source": random.choice([
                    "SIEM Alert System",
                    "Intrusion Detection System",
                    "User Report",
                    "Threat Intelligence Feed"
                ]),
                "breach_type": random.choice([
                    "unauthorized data access",
                    "malware detection",
                    "privilege escalation",
                    "data exfiltration attempt"
                ]),
                "affected_component": random.choice([
                    "customer database",
                    "employee workstation",
                    "web application",
                    "file server"
                ]),
                "discovery_method": random.choice([
                    "automated security monitoring detected unusual access patterns",
                    "an employee reported suspicious activity",
                    "routine security audit identified anomalies"
                ]),
                "initial_assessment": random.choice([
                    "the threat actor may have gained unauthorized access to sensitive data",
                    "malicious software was deployed on internal systems",
                    "credentials may have been compromised"
                ]),
                "primary_system": random.choice([
                    "Customer Database Server (DB-PROD-01)",
                    "Web Application (APP-PROD-02)",
                    "Employee Workstation (WS-FINANCE-15)"
                ]),
                "secondary_systems": "Authentication Server, Backup Systems",
                "compromised_systems": "Under investigation",
                "indicator1": "Unusual login patterns from foreign IP addresses",
                "indicator2": "Elevated privilege escalation attempts",
                "indicator3": "Suspicious file access patterns",
                "indicator4": "Abnormal network traffic volumes",
                "data_type1": "Customer personal information",
                "record_count1": f"{random.randint(1, 50)},000",
                "data_type2": "Payment information",
                "record_count2": f"{random.randint(500, 5000)}",
                "data_type3": "Employee records",
                "record_count3": f"{random.randint(100, 1000)}",
                "data_type4": "System configuration files",
                "containment1": "Isolated affected systems from network",
                "time1": (current_time - timedelta(minutes=30)).strftime("%I:%M %p"),
                "containment2": "Reset potentially compromised credentials",
                "time2": (current_time - timedelta(minutes=15)).strftime("%I:%M %p"),
                "containment3": "Deploy additional monitoring on all systems",
                "time3": (current_time + timedelta(minutes=30)).strftime("%I:%M %p"),
                "containment4": "Engage third-party forensics team",
                "time4": (current_time + timedelta(hours=2)).strftime("%I:%M %p"),
                "sec_lead": random.choice(self.names),
                "sec_contact": f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "legal_contact": random.choice(self.names),
                "legal_phone": f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "comm_contact": random.choice(self.names),
                "compliance_contact": random.choice(self.names),
                "compliance_phone": f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "urgent_action1": "Complete forensic imaging of affected systems",
                "urgent_owner1": random.choice(self.names),
                "urgent_action2": "Conduct threat hunting across all systems",
                "urgent_owner2": random.choice(self.names),
                "urgent_due2": (current_time + timedelta(hours=4)).strftime("%I:%M %p"),
                "urgent_action3": "Prepare customer notification templates",
                "urgent_owner3": random.choice(self.names),
                "urgent_due3": (current_time + timedelta(hours=6)).strftime("%I:%M %p"),
                "urgent_action4": "Coordinate with external forensics team",
                "urgent_owner4": random.choice(self.names),
                "urgent_due4": (current_time + timedelta(hours=8)).strftime("%I:%M %p"),
                "regulator1": "State Attorney General",
                "notification_period1": "72 hours",
                "regulator2": "Federal Trade Commission",
                "notification_period2": "5 business days",
                "customer_notification": "72 hours",
                "exec_time": (current_time + timedelta(hours=2)).strftime("%I:%M %p"),
                "forensic_system1": "DB-PROD-01 (Customer Database)",
                "forensic_system2": "WS-FINANCE-15 (Workstation)",
                "forensic_system3": "LOG-SERVER-01 (Audit Logs)",
                "secure_number": f"1-800-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                "secure_code": random.randint(100000, 999999),
                "backup_number": f"1-800-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
            }
            
            for key, value in variables.items():
                content = content.replace(f"{{{key}}}", str(value))
            
            full_email = f"From: {header['from']}\nDate: {header['date']}\nSubject: {template['subject']}\n\n{content}{signature}"
            
            urgent_emails.append({
                "email": full_email,
                "category": "Urgent",
                "content_length": len(full_email)
            })
        
        return urgent_emails
    
    def generate_standard_emails(self, count: int) -> List[Dict]:
        """Generate realistic standard emails with long format."""
        standard_emails = []
        
        standard_templates = [
            {
                "subject": "Monthly Project Status Report - {month} {year}",
                "content": """Dear Stakeholders,

I hope this email finds you well. Please find below the comprehensive monthly status report for all active projects under our portfolio for {month} {year}.

EXECUTIVE SUMMARY:
This month has been productive with significant progress across multiple project streams. We have {completed_projects} projects reaching completion, {on_track_projects} projects proceeding as planned, and {delayed_projects} projects requiring attention due to various challenges.

Overall portfolio health: {portfolio_health}
Budget utilization: {budget_utilization}%
Resource allocation: {resource_allocation}% capacity

PROJECT STATUS BREAKDOWN:

🟢 PROJECTS ON TRACK ({on_track_count}):

Project Alpha - {project_alpha_name}
• Current Phase: {alpha_phase}
• Progress: {alpha_progress}% complete
• Next Milestone: {alpha_milestone} ({alpha_date})
• Budget Status: {alpha_budget}% utilized
• Key Achievements: {alpha_achievements}
• Team: {alpha_team_size} members

Project Beta - {project_beta_name}
• Current Phase: {beta_phase}
• Progress: {beta_progress}% complete
• Next Milestone: {beta_milestone} ({beta_date})
• Budget Status: {beta_budget}% utilized
• Key Achievements: {beta_achievements}
• Team: {beta_team_size} members

🟡 PROJECTS REQUIRING ATTENTION ({attention_count}):

Project Gamma - {project_gamma_name}
• Current Phase: {gamma_phase}
• Progress: {gamma_progress}% complete
• Issue: {gamma_issue}
• Impact: {gamma_impact}
• Mitigation Plan: {gamma_mitigation}
• Revised Timeline: {gamma_timeline}

🔴 PROJECTS AT RISK ({risk_count}):

Project Delta - {project_delta_name}
• Current Phase: {delta_phase}
• Progress: {delta_progress}% complete
• Risk Factors: {delta_risks}
• Business Impact: {delta_impact}
• Recovery Plan: {delta_recovery}
• Escalation Required: {delta_escalation}

RESOURCE ALLOCATION UPDATE:
• Engineering: {eng_allocation}% capacity ({eng_people} people)
• Design: {design_allocation}% capacity ({design_people} people)
• QA: {qa_allocation}% capacity ({qa_people} people)
• Project Management: {pm_allocation}% capacity ({pm_people} people)

BUDGET SUMMARY:
• Total Allocated Budget: ${total_budget}
• Spent to Date: ${spent_budget} ({spent_percentage}%)
• Remaining Budget: ${remaining_budget}
• Forecasted Overage: ${overage_forecast}
• Cost per Project Average: ${cost_per_project}

KEY ACHIEVEMENTS THIS MONTH:
✅ {achievement1}
✅ {achievement2}
✅ {achievement3}
✅ {achievement4}

CHALLENGES AND RISKS:
⚠️ {challenge1}
⚠️ {challenge2}
⚠️ {challenge3}

UPCOMING MILESTONES (Next 30 Days):
📅 {upcoming1} - {upcoming1_date}
📅 {upcoming2} - {upcoming2_date}
📅 {upcoming3} - {upcoming3_date}

RESOURCE REQUESTS:
Based on current project demands, we are requesting:
• {resource_request1}
• {resource_request2}
• {resource_request3}

QUALITY METRICS:
• Code Review Coverage: {code_coverage}%
• Bug Escape Rate: {bug_rate}%
• Customer Satisfaction: {customer_satisfaction}/10
• Team Velocity: {team_velocity} story points/sprint

NEXT MONTH'S FOCUS:
Our primary objectives for {next_month} include:
1. {next_objective1}
2. {next_objective2}
3. {next_objective3}
4. {next_objective4}

STAKEHOLDER ACTIONS REQUIRED:
• {action_required1} (Due: {action_due1})
• {action_required2} (Due: {action_due2})
• {action_required3} (Due: {action_due3})

Please review this report and let me know if you have any questions or require additional details on any specific project. I'm available for individual discussions if needed.

The detailed project documentation and financial reports are available in our project management system. Access credentials have been shared separately.

Thank you for your continued support and leadership. I look forward to our monthly review meeting scheduled for {review_meeting_date}.
"""
            },
            {
                "subject": "Quarterly Business Review - Vendor Performance Assessment",
                "content": """Dear Procurement Committee,

As part of our quarterly business review process, I am pleased to provide you with a comprehensive assessment of our vendor performance for Q{quarter} {year}.

OVERVIEW:
This quarter, we evaluated {vendor_count} active vendors across {category_count} service categories. The assessment was conducted using our standard vendor scorecard methodology, which evaluates performance across five key dimensions: Quality, Delivery, Cost, Service, and Innovation.

EXECUTIVE SUMMARY:
Overall vendor performance this quarter has been {overall_performance}, with {top_performers} vendors exceeding expectations and {underperformers} requiring performance improvement plans.

Total vendor spend: ${total_spend}
Cost savings achieved: ${cost_savings} ({savings_percentage}%)
Contract renewals processed: {renewals}
New vendor onboarding: {new_vendors}

VENDOR PERFORMANCE SCORECARD:

🥇 TOP PERFORMING VENDORS (Score 90-100):

{vendor1_name} - {vendor1_service}
• Overall Score: {vendor1_score}/100
• Strengths: {vendor1_strengths}
• Contract Value: ${vendor1_value}
• Relationship Duration: {vendor1_duration}
• Recommendation: {vendor1_recommendation}

{vendor2_name} - {vendor2_service}
• Overall Score: {vendor2_score}/100
• Strengths: {vendor2_strengths}
• Contract Value: ${vendor2_value}
• Relationship Duration: {vendor2_duration}
• Recommendation: {vendor2_recommendation}

🟡 AVERAGE PERFORMING VENDORS (Score 70-89):

{vendor3_name} - {vendor3_service}
• Overall Score: {vendor3_score}/100
• Areas for Improvement: {vendor3_improvements}
• Contract Value: ${vendor3_value}
• Action Plan: {vendor3_action}

🔴 UNDERPERFORMING VENDORS (Score Below 70):

{vendor4_name} - {vendor4_service}
• Overall Score: {vendor4_score}/100
• Critical Issues: {vendor4_issues}
• Contract Value: ${vendor4_value}
• Performance Improvement Plan: {vendor4_pip}
• Review Timeline: {vendor4_timeline}

CATEGORY ANALYSIS:

IT Services & Software:
• Vendor Count: {it_vendor_count}
• Average Score: {it_avg_score}/100
• Total Spend: ${it_spend}
• Key Trends: {it_trends}

Professional Services:
• Vendor Count: {prof_vendor_count}
• Average Score: {prof_avg_score}/100
• Total Spend: ${prof_spend}
• Key Trends: {prof_trends}

Facilities & Maintenance:
• Vendor Count: {facilities_vendor_count}
• Average Score: {facilities_avg_score}/100
• Total Spend: ${facilities_spend}
• Key Trends: {facilities_trends}

COST MANAGEMENT ACHIEVEMENTS:
💰 Successfully renegotiated {renegotiated_contracts} contracts
💰 Achieved ${volume_savings} in volume discounts
💰 Consolidated {consolidated_vendors} vendors for better pricing
💰 Implemented {cost_initiatives} cost optimization initiatives

KEY PERFORMANCE INDICATORS:

Delivery Performance:
• On-time delivery rate: {delivery_rate}%
• Quality defect rate: {defect_rate}%
• SLA compliance: {sla_compliance}%

Financial Performance:
• Invoice accuracy: {invoice_accuracy}%
• Payment disputes: {payment_disputes}
• Cost variance from budget: {cost_variance}%

Service Level Performance:
• Response time to issues: {response_time} hours average
• Resolution time: {resolution_time} hours average
• Customer satisfaction: {vendor_satisfaction}/10

RISKS AND CONCERNS:
🚨 {risk1}
🚨 {risk2}
🚨 {risk3}

VENDOR DEVELOPMENT INITIATIVES:
This quarter, we launched several vendor development programs:
• {initiative1}
• {initiative2}
• {initiative3}

UPCOMING CONTRACT RENEWALS (Next Quarter):
📋 {renewal1} - Contract value: ${renewal1_value} (Expires: {renewal1_date})
📋 {renewal2} - Contract value: ${renewal2_value} (Expires: {renewal2_date})
📋 {renewal3} - Contract value: ${renewal3_value} (Expires: {renewal3_date})

RECOMMENDATIONS FOR NEXT QUARTER:
1. {recommendation1}
2. {recommendation2}
3. {recommendation3}
4. {recommendation4}

PROCUREMENT STRATEGY UPDATES:
Based on this quarter's performance analysis, we recommend:
• {strategy_update1}
• {strategy_update2}
• {strategy_update3}

COMPLIANCE AND AUDIT STATUS:
• Vendor compliance audits completed: {compliance_audits}
• Compliance violations identified: {compliance_violations}
• Corrective actions implemented: {corrective_actions}
• Certification renewals: {cert_renewals}

The complete vendor scorecards, detailed performance metrics, and supporting documentation are available in the procurement portal. Individual vendor performance discussions can be scheduled upon request.

Please review this assessment and provide any feedback or questions by {feedback_deadline}. Our next quarterly review meeting is scheduled for {next_review_date}.

Thank you for your continued partnership in maintaining our vendor excellence standards.
"""
            }
        ]
        
        for i in range(count):
            template = random.choice(standard_templates)
            header = self.generate_email_header("Standard")
            signature = self.generate_email_signature("Work", header["sender_name"])
            
            # Fill template variables
            content = template["content"]
            current_date = datetime.now()
            
            variables = {
                "month": current_date.strftime("%B"),
                "year": current_date.year,
                "completed_projects": random.randint(2, 5),
                "on_track_projects": random.randint(5, 12),
                "delayed_projects": random.randint(1, 3),
                "portfolio_health": random.choice(["Strong", "Good", "Moderate", "Needs Attention"]),
                "budget_utilization": random.randint(65, 95),
                "resource_allocation": random.randint(80, 100),
                "on_track_count": random.randint(3, 8),
                "project_alpha_name": random.choice([
                    "Customer Portal Redesign",
                    "Mobile App Development",
                    "Data Migration Project",
                    "Security Enhancement Initiative"
                ]),
                "alpha_phase": random.choice(["Development", "Testing", "Deployment", "Design"]),
                "alpha_progress": random.randint(60, 85),
                "alpha_milestone": random.choice([
                    "User Acceptance Testing",
                    "Production Deployment",
                    "Feature Complete",
                    "Beta Release"
                ]),
                "alpha_date": (current_date + timedelta(days=random.randint(7, 30))).strftime("%B %d"),
                "alpha_budget": random.randint(70, 90),
                "alpha_achievements": random.choice([
                    "Completed user interface design phase",
                    "Successfully integrated payment gateway",
                    "Achieved 99.9% uptime in testing environment"
                ]),
                "alpha_team_size": random.randint(5, 12),
                "project_beta_name": random.choice([
                    "ERP System Upgrade",
                    "Cloud Migration",
                    "Business Intelligence Platform",
                    "Workflow Automation"
                ]),
                "beta_phase": random.choice(["Planning", "Development", "Integration", "Testing"]),
                "beta_progress": random.randint(45, 75),
                "beta_milestone": random.choice([
                    "System Integration Testing",
                    "Data Validation Complete",
                    "Performance Testing",
                    "Go-Live Preparation"
                ]),
                "beta_date": (current_date + timedelta(days=random.randint(14, 45))).strftime("%B %d"),
                "beta_budget": random.randint(65, 85),
                "beta_achievements": random.choice([
                    "Completed database schema migration",
                    "Established CI/CD pipeline",
                    "Completed security audit with zero critical findings"
                ]),
                "beta_team_size": random.randint(4, 10),
                "attention_count": random.randint(1, 3),
                "project_gamma_name": random.choice([
                    "Legacy System Modernization",
                    "API Development Platform",
                    "Customer Data Platform"
                ]),
                "gamma_phase": random.choice(["Development", "Testing", "Integration"]),
                "gamma_progress": random.randint(35, 65),
                "gamma_issue": random.choice([
                    "Resource availability constraints due to competing priorities",
                    "Technical complexity higher than initially estimated",
                    "Third-party vendor delays affecting timeline"
                ]),
                "gamma_impact": random.choice([
                    "2-week delay to original timeline",
                    "15% budget increase required",
                    "Scope reduction may be necessary"
                ]),
                "gamma_mitigation": random.choice([
                    "Additional contractor resources secured",
                    "Parallel development streams established",
                    "Vendor escalation meeting scheduled"
                ]),
                "gamma_timeline": (current_date + timedelta(days=random.randint(30, 60))).strftime("%B %d"),
                "risk_count": random.randint(0, 2),
                "project_delta_name": random.choice([
                    "Digital Transformation Initiative",
                    "Customer Experience Platform",
                    "Advanced Analytics Solution"
                ]),
                "delta_phase": random.choice(["Requirements", "Design", "Development"]),
                "delta_progress": random.randint(15, 45),
                "delta_risks": random.choice([
                    "Key technical lead departure, knowledge transfer incomplete",
                    "Integration complexity with legacy systems underestimated",
                    "Budget constraints affecting resource allocation"
                ]),
                "delta_impact": random.choice([
                    "Potential 6-week delay to project timeline",
                    "30% budget overrun projected",
                    "Quality deliverables may be compromised"
                ]),
                "delta_recovery": random.choice([
                    "Emergency hiring of senior technical resources",
                    "Phased delivery approach to reduce risk",
                    "Executive committee review scheduled"
                ]),
                "delta_escalation": random.choice([
                    "Yes - VP Engineering involvement required",
                    "Yes - Budget committee approval needed",
                    "No - team-level resolution in progress"
                ]),
                "eng_allocation": random.randint(85, 100),
                "eng_people": random.randint(15, 30),
                "design_allocation": random.randint(75, 95),
                "design_people": random.randint(5, 12),
                "qa_allocation": random.randint(80, 100),
                "qa_people": random.randint(8, 15),
                "pm_allocation": random.randint(90, 100),
                "pm_people": random.randint(6, 12),
                "total_budget": f"{random.randint(500, 2000)},000",
                "spent_budget": f"{random.randint(300, 800)},000",
                "spent_percentage": random.randint(60, 80),
                "remaining_budget": f"{random.randint(200, 700)},000",
                "overage_forecast": f"{random.randint(0, 100)},000",
                "cost_per_project": f"{random.randint(50, 200)},000",
                "achievement1": random.choice([
                    "Completed user research phase for 3 major projects",
                    "Successfully deployed security updates across all systems",
                    "Achieved 99.8% system uptime across all platforms"
                ]),
                "achievement2": random.choice([
                    "Reduced average bug resolution time by 35%",
                    "Completed training program for 25 team members",
                    "Implemented automated testing reducing manual effort by 40%"
                ]),
                "achievement3": random.choice([
                    "Launched customer feedback portal with 95% satisfaction",
                    "Optimized database performance improving response time by 50%",
                    "Established new vendor partnerships reducing costs by 20%"
                ]),
                "achievement4": random.choice([
                    "Completed annual security compliance audit successfully",
                    "Migrated 80% of workloads to cloud infrastructure",
                    "Delivered 3 major feature releases ahead of schedule"
                ]),
                "challenge1": random.choice([
                    "Talent acquisition challenges in specialized technical roles",
                    "Integration complexity between new and legacy systems",
                    "Vendor delays affecting multiple project timelines"
                ]),
                "challenge2": random.choice([
                    "Budget constraints requiring prioritization of initiatives",
                    "Remote work coordination across multiple time zones",
                    "Rapidly changing business requirements affecting scope"
                ]),
                "challenge3": random.choice([
                    "Technical debt accumulation requiring dedicated remediation",
                    "Regulatory compliance requirements adding complexity",
                    "Market competition requiring accelerated delivery timelines"
                ]),
                "upcoming1": random.choice([
                    "Project Alpha Beta Release",
                    "Security Audit Completion",
                    "Vendor Contract Renewal"
                ]),
                "upcoming1_date": (current_date + timedelta(days=random.randint(7, 21))).strftime("%B %d"),
                "upcoming2": random.choice([
                    "Customer Portal Go-Live",
                    "Data Migration Completion",
                    "Performance Testing Phase"
                ]),
                "upcoming2_date": (current_date + timedelta(days=random.randint(14, 35))).strftime("%B %d"),
                "upcoming3": random.choice([
                    "Quarterly Business Review",
                    "Team Training Completion",
                    "Infrastructure Upgrade"
                ]),
                "upcoming3_date": (current_date + timedelta(days=random.randint(21, 42))).strftime("%B %d"),
                "resource_request1": random.choice([
                    "2 additional senior developers for Q4 projects",
                    "UX designer contractor for 3-month engagement",
                    "DevOps engineer to support infrastructure scaling"
                ]),
                "resource_request2": random.choice([
                    "Additional testing environment for parallel development",
                    "Project management tool licenses for expanding team",
                    "Cloud infrastructure budget increase for growth"
                ]),
                "resource_request3": random.choice([
                    "Training budget for team certification programs",
                    "Third-party security assessment services",
                    "Vendor support for critical system maintenance"
                ]),
                "code_coverage": random.randint(75, 95),
                "bug_rate": random.randint(1, 8),
                "customer_satisfaction": random.randint(7, 10),
                "team_velocity": random.randint(35, 65),
                "next_month": (current_date + timedelta(days=30)).strftime("%B"),
                "next_objective1": random.choice([
                    "Complete mobile app development and begin testing phase",
                    "Finalize vendor selections for infrastructure upgrades",
                    "Launch customer feedback collection system"
                ]),
                "next_objective2": random.choice([
                    "Implement automated deployment pipeline for all projects",
                    "Complete security compliance audit preparations",
                    "Establish cross-functional collaboration frameworks"
                ]),
                "next_objective3": random.choice([
                    "Reduce technical debt by 25% across legacy systems",
                    "Complete team capacity planning for next quarter",
                    "Launch internal developer productivity initiative"
                ]),
                "next_objective4": random.choice([
                    "Establish performance monitoring for all applications",
                    "Complete documentation updates for all active projects",
                    "Implement cost optimization measures across cloud resources"
                ]),
                "action_required1": random.choice([
                    "Approve additional budget allocation for Project Gamma",
                    "Review and sign vendor contracts for Q4 initiatives",
                    "Provide feedback on proposed timeline adjustments"
                ]),
                "action_due1": (current_date + timedelta(days=random.randint(5, 14))).strftime("%B %d"),
                "action_required2": random.choice([
                    "Schedule stakeholder interviews for requirements gathering",
                    "Approve hiring plan for additional team members",
                    "Review risk mitigation strategies for at-risk projects"
                ]),
                "action_due2": (current_date + timedelta(days=random.randint(7, 21))).strftime("%B %d"),
                "action_required3": random.choice([
                    "Participate in quarterly business review presentation",
                    "Provide input on technology strategy for next year",
                    "Review and approve change management procedures"
                ]),
                "action_due3": (current_date + timedelta(days=random.randint(14, 28))).strftime("%B %d"),
                "review_meeting_date": (current_date + timedelta(days=random.randint(7, 14))).strftime("%B %d"),
                # Vendor assessment variables
                "quarter": random.randint(1, 4),
                "vendor_count": random.randint(25, 50),
                "category_count": random.randint(8, 15),
                "overall_performance": random.choice(["excellent", "good", "satisfactory", "mixed"]),
                "top_performers": random.randint(5, 12),
                "underperformers": random.randint(2, 6),
                "total_spend": f"{random.randint(2, 20)}.{random.randint(1, 9)}M",
                "cost_savings": f"{random.randint(100, 800)}K",
                "savings_percentage": random.randint(5, 25),
                "renewals": random.randint(8, 20),
                "new_vendors": random.randint(3, 10),
                "vendor1_name": random.choice([
                    "TechSolutions Inc", "Global Services Corp", "Innovation Partners LLC",
                    "Enterprise Systems Ltd", "Digital Consulting Group"
                ]),
                "vendor1_service": random.choice([
                    "Cloud Infrastructure Services", "Software Development", "IT Support Services",
                    "Data Analytics Platform", "Cybersecurity Solutions"
                ]),
                "vendor1_score": random.randint(90, 98),
                "vendor1_strengths": random.choice([
                    "Exceptional response times, proactive communication",
                    "Innovative solutions, cost-effective pricing",
                    "High quality deliverables, experienced team"
                ]),
                "vendor1_value": f"{random.randint(500, 2000)}K",
                "vendor1_duration": f"{random.randint(2, 8)} years",
                "vendor1_recommendation": random.choice([
                    "Extend contract for additional 2 years",
                    "Expand scope to include additional services",
                    "Consider for strategic partnership status"
                ]),
                "vendor2_name": random.choice([
                    "Professional Services Inc", "Strategic Consulting", "Business Solutions Ltd",
                    "Technology Partners", "Excellence Corp"
                ]),
                "vendor2_service": random.choice([
                    "Business Process Consulting", "ERP Implementation", "Change Management",
                    "Training and Development", "Quality Assurance"
                ]),
                "vendor2_score": random.randint(91, 97),
                "vendor2_strengths": random.choice([
                    "Domain expertise, flexible engagement model",
                    "Consistent delivery, excellent project management",
                    "Cultural fit, innovative approach"
                ]),
                "vendor2_value": f"{random.randint(300, 1500)}K",
                "vendor2_duration": f"{random.randint(1, 6)} years",
                "vendor2_recommendation": random.choice([
                    "Renew contract with expanded scope",
                    "Consider for additional project opportunities",
                    "Maintain current engagement level"
                ]),
                "vendor3_name": random.choice([
                    "Standard Solutions", "Regional Services", "Mid-Tier Consulting",
                    "Local Partners LLC", "Basic Services Inc"
                ]),
                "vendor3_service": random.choice([
                    "Facilities Management", "Equipment Maintenance", "Help Desk Services",
                    "Document Management", "General Contracting"
                ]),
                "vendor3_score": random.randint(70, 89),
                "vendor3_improvements": random.choice([
                    "Response time improvement needed",
                    "Better project communication required",
                    "Quality consistency needs attention"
                ]),
                "vendor3_value": f"{random.randint(100, 800)}K",
                "vendor3_action": random.choice([
                    "Monthly performance reviews implemented",
                    "Service level agreement updates",
                    "Additional training requirements"
                ]),
                "vendor4_name": random.choice([
                    "Underperforming LLC", "Challenged Services", "Problematic Corp",
                    "Issues Inc", "Struggling Solutions"
                ]),
                "vendor4_service": random.choice([
                    "Legacy System Maintenance", "Third-Party Integration", "Specialized Consulting",
                    "Niche Software Services", "Custom Development"
                ]),
                "vendor4_score": random.randint(45, 69),
                "vendor4_issues": random.choice([
                    "Consistent delivery delays, quality issues",
                    "Poor communication, missed deadlines",
                    "Resource turnover, lack of expertise"
                ]),
                "vendor4_value": f"{random.randint(50, 500)}K",
                "vendor4_pip": random.choice([
                    "90-day improvement plan with weekly checkpoints",
                    "Immediate escalation process implementation",
                    "Contract amendment with penalty clauses"
                ]),
                "vendor4_timeline": f"{random.randint(60, 120)} days",
                "it_vendor_count": random.randint(8, 15),
                "it_avg_score": random.randint(75, 90),
                "it_spend": f"{random.randint(3, 12)}M",
                "it_trends": random.choice([
                    "Increased focus on cloud services and automation",
                    "Growing emphasis on cybersecurity solutions",
                    "Shift toward managed services model"
                ]),
                "prof_vendor_count": random.randint(5, 12),
                "prof_avg_score": random.randint(70, 85),
                "prof_spend": f"{random.randint(1, 6)}M",
                "prof_trends": random.choice([
                    "Demand for specialized digital transformation expertise",
                    "Increased focus on change management capabilities",
                    "Growing need for agile project delivery methods"
                ]),
                "facilities_vendor_count": random.randint(6, 10),
                "facilities_avg_score": random.randint(65, 80),
                "facilities_spend": f"{random.randint(500, 3000)}K",
                "facilities_trends": random.choice([
                    "Emphasis on sustainable and energy-efficient solutions",
                    "Integration of smart building technologies",
                    "Flexible space management for hybrid work models"
                ]),
                "renegotiated_contracts": random.randint(5, 15),
                "volume_savings": f"{random.randint(50, 300)}K",
                "consolidated_vendors": random.randint(3, 8),
                "cost_initiatives": random.randint(4, 12),
                "delivery_rate": random.randint(85, 98),
                "defect_rate": random.randint(1, 8),
                "sla_compliance": random.randint(90, 99),
                "invoice_accuracy": random.randint(92, 99),
                "payment_disputes": random.randint(0, 5),
                "cost_variance": random.randint(-5, 15),
                "response_time": random.randint(2, 24),
                "resolution_time": random.randint(8, 72),
                "vendor_satisfaction": random.randint(6, 10),
                "risk1": random.choice([
                    "Single source dependency for critical services",
                    "Potential vendor financial instability concerns",
                    "Lack of backup vendors for specialized services"
                ]),
                "risk2": random.choice([
                    "Contract renewal negotiations for major vendors",
                    "Skill gap in emerging technology requirements",
                    "Regulatory compliance requirements affecting vendor selection"
                ]),
                "risk3": random.choice([
                    "Market consolidation reducing vendor options",
                    "Rising costs across multiple service categories",
                    "Geographic concentration of key vendor operations"
                ]),
                "initiative1": random.choice([
                    "Vendor innovation workshops and capability assessments",
                    "Joint cost reduction initiatives and process improvements",
                    "Diversity and inclusion vendor certification program"
                ]),
                "initiative2": random.choice([
                    "Quarterly vendor performance review meetings",
                    "Technology roadmap collaboration sessions",
                    "Vendor relationship management training for procurement team"
                ]),
                "initiative3": random.choice([
                    "Sustainability and environmental impact assessment program",
                    "Vendor risk assessment and business continuity planning",
                    "Digital transformation partnership development"
                ]),
                "renewal1": random.choice([
                    "Primary IT Infrastructure Support",
                    "Enterprise Software Licensing",
                    "Professional Services Framework"
                ]),
                "renewal1_value": f"{random.randint(500, 2500)}K",
                "renewal1_date": (current_date + timedelta(days=random.randint(30, 90))).strftime("%B %d, %Y"),
                "renewal2": random.choice([
                    "Cloud Platform Services",
                    "Cybersecurity Managed Services",
                    "Business Process Outsourcing"
                ]),
                "renewal2_value": f"{random.randint(300, 1800)}K",
                "renewal2_date": (current_date + timedelta(days=random.randint(45, 120))).strftime("%B %d, %Y"),
                "renewal3": random.choice([
                    "Facilities Management Contract",
                    "Legal Services Retainer",
                    "Marketing Agency Partnership"
                ]),
                "renewal3_value": f"{random.randint(200, 1200)}K",
                "renewal3_date": (current_date + timedelta(days=random.randint(60, 150))).strftime("%B %d, %Y"),
                "recommendation1": random.choice([
                    "Implement vendor scoreboard dashboard for real-time monitoring",
                    "Establish preferred vendor program with tiered benefits",
                    "Develop vendor innovation fund for joint R&D initiatives"
                ]),
                "recommendation2": random.choice([
                    "Conduct market analysis for competitive benchmarking",
                    "Implement automated invoice processing to reduce disputes",
                    "Establish vendor diversity targets and measurement criteria"
                ]),
                "recommendation3": random.choice([
                    "Create vendor risk management framework and monitoring system",
                    "Develop long-term strategic partnerships with top performers",
                    "Implement sustainability requirements in all vendor contracts"
                ]),
                "recommendation4": random.choice([
                    "Enhance vendor onboarding process with digital tools",
                    "Establish center of excellence for vendor management best practices",
                    "Implement predictive analytics for vendor performance forecasting"
                ]),
                "strategy_update1": random.choice([
                    "Shift toward outcome-based contracting models",
                    "Increase emphasis on vendor innovation capabilities",
                    "Prioritize vendors with strong ESG credentials"
                ]),
                "strategy_update2": random.choice([
                    "Implement category management approach for strategic sourcing",
                    "Develop multi-year framework agreements for stability",
                    "Establish vendor relationship tiers with differentiated management"
                ]),
                "strategy_update3": random.choice([
                    "Enhance vendor selection criteria to include digital capabilities",
                    "Implement total cost of ownership analysis for all major contracts",
                    "Develop contingency plans for critical vendor dependencies"
                ]),
                "compliance_audits": random.randint(8, 20),
                "compliance_violations": random.randint(0, 5),
                "corrective_actions": random.randint(2, 12),
                "cert_renewals": random.randint(5, 15),
                "feedback_deadline": (current_date + timedelta(days=random.randint(7, 14))).strftime("%B %d"),
                "next_review_date": (current_date + timedelta(days=random.randint(21, 35))).strftime("%B %d, %Y")
            }
            
            for key, value in variables.items():
                content = content.replace(f"{{{key}}}", str(value))
            
            full_email = f"From: {header['from']}\nDate: {header['date']}\nSubject: {template['subject']}\n\n{content}{signature}"
            
            standard_emails.append({
                "email": full_email,
                "category": "Standard",
                "content_length": len(full_email)
            })
        
        return standard_emails
    
    def generate_comprehensive_dataset(self, 
                                     spam_count: int = 1000, 
                                     work_count: int = 1000, 
                                     personal_count: int = 1000, 
                                     urgent_count: int = 1000, 
                                     standard_count: int = 1000) -> pd.DataFrame:
        """Generate comprehensive long-format email dataset."""
        
        print(f"🎯 GENERATING COMPREHENSIVE LONG-FORMAT EMAIL DATASET")
        print("=" * 60)
        
        all_emails = []
        
        # Generate each category
        print(f"📧 Generating {spam_count} spam emails...")
        all_emails.extend(self.generate_spam_emails(spam_count))
        
        print(f"💼 Generating {work_count} work emails...")
        all_emails.extend(self.generate_work_emails(work_count))
        
        print(f"👥 Generating {personal_count} personal emails...")
        all_emails.extend(self.generate_personal_emails(personal_count))
        
        print(f"🚨 Generating {urgent_count} urgent emails...")
        all_emails.extend(self.generate_urgent_emails(urgent_count))
        
        print(f"📋 Generating {standard_count} standard emails...")
        all_emails.extend(self.generate_standard_emails(standard_count))
        
        # Create DataFrame
        df = pd.DataFrame(all_emails)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate statistics
        avg_length = df['content_length'].mean()
        min_length = df['content_length'].min()
        max_length = df['content_length'].max()
        
        print(f"\n✅ DATASET GENERATION COMPLETE")
        print("=" * 60)
        print(f"Total emails generated: {len(df):,}")
        print(f"Average email length: {avg_length:.0f} characters")
        print(f"Shortest email: {min_length:,} characters")
        print(f"Longest email: {max_length:,} characters")
        print(f"\nCategory distribution:")
        
        for category, count in df['category'].value_counts().items():
            percentage = (count / len(df)) * 100
            print(f"  {category}: {count:,} emails ({percentage:.1f}%)")
        
        return df


def main():
    """Generate sample dataset."""
    generator = EnhancedEmailDatasetGenerator()
    
    # Generate smaller sample for testing
    df = generator.generate_comprehensive_dataset(
        spam_count=10,
        work_count=10, 
        personal_count=10,
        urgent_count=10,
        standard_count=10
    )
    
    # Save sample
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enhanced_email_dataset_sample_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n💾 Sample dataset saved to: {filename}")
    
    # Show sample emails
    print(f"\n📖 SAMPLE EMAILS")
    print("=" * 60)
    
    for category in df['category'].unique():
        sample_email = df[df['category'] == category].iloc[0]['email']
        print(f"\n{category.upper()} EMAIL SAMPLE:")
        print("-" * 40)
        print(sample_email[:500] + "..." if len(sample_email) > 500 else sample_email)


if __name__ == "__main__":
    main()
