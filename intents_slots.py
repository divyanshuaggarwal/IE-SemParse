itop_intents = ['IN:GET_MESSAGE',
 'IN:GET_WEATHER',
 'IN:GET_ALARM',
 'IN:SEND_MESSAGE',
 'IN:GET_INFO_RECIPES',
 'IN:SET_UNAVAILABLE',
 'IN:DELETE_REMINDER',
 'IN:GET_STORIES_NEWS',
 'IN:CREATE_ALARM',
 'IN:GET_REMINDER',
 'IN:CREATE_REMINDER',
 'IN:GET_RECIPES',
 'IN:QUESTION_NEWS',
 'IN:GET_EVENT',
 'IN:PLAY_MUSIC',
 'IN:GET_CALL_TIME',
 'IN:CREATE_CALL',
 'IN:END_CALL',
 'IN:CREATE_PLAYLIST_MUSIC',
 'IN:CREATE_TIMER',
 'IN:IGNORE_CALL',
 'IN:GET_LIFE_EVENT',
 'IN:GET_INFO_CONTACT',
 'IN:UPDATE_CALL',
 'IN:UPDATE_REMINDER_DATE_TIME',
 'IN:GET_CONTACT',
 'IN:GET_TIMER',
 'IN:GET_REMINDER_DATE_TIME',
 'IN:DELETE_ALARM',
 'IN:PAUSE_MUSIC',
 'IN:GET_AGE',
 'IN:GET_SUNRISE',
 'IN:GET_EMPLOYER',
 'IN:GET_EDUCATION_TIME',
 'IN:ANSWER_CALL',
 'IN:SET_RSVP_YES',
 'IN:SNOOZE_ALARM',
 'IN:GET_JOB',
 'IN:UPDATE_REMINDER_TODO',
 'IN:IS_TRUE_RECIPES',
 'IN:REMOVE_FROM_PLAYLIST_MUSIC',
 'IN:GET_AVAILABILITY',
 'IN:GET_CATEGORY_EVENT',
 'IN:PLAY_MEDIA',
 'IN:ADD_TIME_TIMER',
 'IN:GET_CALL',
 'IN:SET_AVAILABLE',
 'IN:ADD_TO_PLAYLIST_MUSIC',
 'IN:GET_EMPLOYMENT_TIME',
 'IN:SHARE_EVENT',
 'IN:PREFER',
 'IN:START_SHUFFLE_MUSIC',
 'IN:GET_CALL_CONTACT',
 'IN:GET_LOCATION',
 'IN:SILENCE_ALARM',
 'IN:SWITCH_CALL',
 'IN:GET_TRACK_INFO_MUSIC',
 'IN:SUBTRACT_TIME_TIMER',
 'IN:GET_SUNSET',
 'IN:DELETE_TIMER',
 'IN:UPDATE_TIMER',
 'IN:PREVIOUS_TRACK_MUSIC',
 'IN:SET_DEFAULT_PROVIDER_MUSIC',
 'IN:HOLD_CALL',
 'IN:GET_MUTUAL_FRIENDS',
 'IN:SKIP_TRACK_MUSIC',
 'IN:UPDATE_METHOD_CALL',
 'IN:SET_RSVP_INTERESTED',
 'IN:QUESTION_MUSIC',
 'IN:GET_UNDERGRAD',
 'IN:PAUSE_TIMER',
 'IN:UPDATE_ALARM',
 'IN:GET_REMINDER_LOCATION',
 'IN:GET_ATTENDEE_EVENT',
 'IN:LIKE_MUSIC',
 'IN:RESTART_TIMER',
 'IN:RESUME_TIMER',
 'IN:MERGE_CALL',
 'IN:GET_MESSAGE_CONTACT',
 'IN:REPLAY_MUSIC',
 'IN:LOOP_MUSIC',
 'IN:GET_REMINDER_AMOUNT',
 'IN:GET_DATE_TIME_EVENT',
 'IN:STOP_MUSIC',
 'IN:GET_DETAILS_NEWS',
 'IN:GET_EDUCATION_DEGREE',
 'IN:SET_DEFAULT_PROVIDER_CALLING',
 'IN:GET_MAJOR',
 'IN:UNLOOP_MUSIC',
 'IN:GET_CONTACT_METHOD',
 'IN:SET_RSVP_NO',
 'IN:UPDATE_REMINDER_LOCATION',
 'IN:RESUME_CALL',
 'IN:CANCEL_MESSAGE',
 'IN:RESUME_MUSIC',
 'IN:UPDATE_REMINDER',
 'IN:DELETE_PLAYLIST_MUSIC',
 'IN:REWIND_MUSIC',
 'IN:REPEAT_ALL_MUSIC',
 'IN:FAST_FORWARD_MUSIC',
 'IN:DISLIKE_MUSIC',
 'IN:GET_LIFE_EVENT_TIME',
 'IN:DISPREFER',
 'IN:REPEAT_ALL_OFF_MUSIC',
 'IN:HELP_REMINDER',
 'IN:GET_LYRICS_MUSIC',
 'IN:STOP_SHUFFLE_MUSIC',
 'IN:GET_AIRQUALITY',
 'IN:GET_LANGUAGE',
 'IN:FOLLOW_MUSIC',
 'IN:GET_GENDER',
 'IN:CANCEL_CALL',
 'IN:GET_GROUP']


itop_slots = ['SL:',
 'SL:AGE',
 'SL:ALARM_NAME',
 'SL:AMOUNT',
 'SL:ATTENDEE',
 'SL:ATTENDEE_EVENT',
 'SL:ATTRIBUTE_EVENT',
 'SL:CATEGORY_EVENT',
 'SL:CONTACT',
 'SL:CONTACT_ADDED',
 'SL:CONTACT_METHOD',
 'SL:CONTACT_RELATED',
 'SL:CONTACT_REMOVED',
 'SL:CONTENT_EXACT',
 'SL:DATE_TIME',
 'SL:EDUCATION_DEGREE',
 'SL:EMPLOYER',
 'SL:GENDER',
 'SL:GROUP',
 'SL:JOB',
 'SL:LIFE_EVENT',
 'SL:LOCATION',
 'SL:MAJOR',
 'SL:METHOD_RECIPES',
 'SL:METHOD_RETRIEVAL_REMINDER',
 'SL:METHOD_TIMER',
 'SL:MUSIC_ALBUM_MODIFIER',
 'SL:MUSIC_ALBUM_TITLE',
 'SL:MUSIC_ARTIST_NAME',
 'SL:MUSIC_GENRE',
 'SL:MUSIC_PLAYLIST_MODIFIER',
 'SL:MUSIC_PLAYLIST_TITLE',
 'SL:MUSIC_PROVIDER_NAME',
 'SL:MUSIC_RADIO_ID',
 'SL:MUSIC_REWIND_TIME',
 'SL:MUSIC_TRACK_TITLE',
 'SL:MUSIC_TYPE',
 'SL:NAME_APP',
 'SL:NEWS_CATEGORY',
 'SL:NEWS_REFERENCE',
 'SL:NEWS_SOURCE',
 'SL:NEWS_TOPIC',
 'SL:NEWS_TYPE',
 'SL:ORDINAL',
 'SL:PERIOD',
 'SL:PERSON_REMINDED',
 'SL:PHONE_NUMBER',
 'SL:RECIPES_ATTRIBUTE',
 'SL:RECIPES_COOKING_METHOD',
 'SL:RECIPES_CUISINE',
 'SL:RECIPES_DIET',
 'SL:RECIPES_DISH',
 'SL:RECIPES_EXCLUDED_INGREDIENT',
 'SL:RECIPES_INCLUDED_INGREDIENT',
 'SL:RECIPES_MEAL',
 'SL:RECIPES_QUALIFIER_NUTRITION',
 'SL:RECIPES_RATING',
 'SL:RECIPES_SOURCE',
 'SL:RECIPES_TYPE',
 'SL:RECIPES_TYPE_NUTRITION',
 'SL:RECIPES_UNIT_MEASUREMENT',
 'SL:RECIPES_UNIT_NUTRITION',
 'SL:RECIPIENT',
 'SL:SCHOOL',
 'SL:SENDER',
 'SL:SIMILARITY',
 'SL:TIMER_NAME',
 'SL:TITLE_EVENT',
 'SL:TODO',
 'SL:TYPE_CONTACT',
 'SL:TYPE_CONTENT',
 'SL:TYPE_RELATION',
 'SL:USER_ATTENDEE_EVENT',
 'SL:WEATHER_ATTRIBUTE',
 'SL:WEATHER_TEMPERATURE_UNIT']


indic_atis_intents = ['IN:FLIGHT',
 'IN:FLIGHT_TIME',
 'IN:AIRFARE',
 'IN:GROUND_SERVICE',
 'IN:AIRPORT',
 'IN:AIRCRAFT',
 'IN:AIRLINE',
 'IN:DISTANCE',
 'IN:GROUND_FARE',
 'IN:ABBREVIATION',
 'IN:QUANTITY',
 'IN:CITY',
 'IN:FLIGHT_NO',
 'IN:CAPACITY',
 'IN:FLIGHT+AIRFARE',
 'IN:MEAL',
 'IN:RESTRICTION',
 'IN:AIRLINE+FLIGHT_NO',
 'IN:GROUND_SERVICE+GROUND_FARE',
 'IN:AIRFARE+FLIGHT_TIME',
 'IN:CHEAPEST',
 'IN:AIRCRAFT+FLIGHT+FLIGHT_NO']

indic_atis_slots = ['SL:AIRCRAFT_CODE',
 'SL:AIRLINE_CODE',
 'SL:AIRLINE_NAME',
 'SL:AIRPORT_CODE',
 'SL:AIRPORT_NAME',
 'SL:CITY_NAME',
 'SL:CLASS_TYPE',
 'SL:CONNECT',
 'SL:COST_RELATIVE',
 'SL:COUNTRY_NAME',
 'SL:DATE_RELATIVE',
 'SL:DAYS_CODE',
 'SL:DAY_NAME',
 'SL:DAY_NUMBER',
 'SL:ECONOMY',
 'SL:END_TIME',
 'SL:FARE_AMOUNT',
 'SL:FARE_BASIS_CODE',
 'SL:FLIGHT_DAYS',
 'SL:FLIGHT_MOD',
 'SL:FLIGHT_NUMBER',
 'SL:FLIGHT_STOP',
 'SL:FLIGHT_TIME',
 'SL:MEAL',
 'SL:MEAL_CODE',
 'SL:MEAL_DESCRIPTION',
 'SL:MOD',
 'SL:MONTH_NAME',
 'SL:OR',
 'SL:PERIOD_MOD',
 'SL:PERIOD_OF_DAY',
 'SL:RESTRICTION_CODE',
 'SL:ROUND_TRIP',
 'SL:START_TIME',
 'SL:STATE_CODE',
 'SL:STATE_NAME',
 'SL:TIME',
 'SL:TIME_RELATIVE',
 'SL:TODAY_RELATIVE',
 'SL:TRANSPORT_TYPE',
 'SL:YEAR']

indic_top_intents = ['IN:COMBINE',
 'IN:GET_CONTACT',
 'IN:GET_DIRECTIONS',
 'IN:GET_DISTANCE',
 'IN:GET_ESTIMATED_ARRIVAL',
 'IN:GET_ESTIMATED_DEPARTURE',
 'IN:GET_ESTIMATED_DURATION',
 'IN:GET_EVENT',
 'IN:GET_EVENT_ATTENDEE',
 'IN:GET_EVENT_ATTENDEE_AMOUNT',
 'IN:GET_EVENT_ORGANIZER',
 'IN:GET_INFO_ROAD_CONDITION',
 'IN:GET_INFO_ROUTE',
 'IN:GET_INFO_TRAFFIC',
 'IN:GET_LOCATION',
 'IN:GET_LOCATION_HOME',
 'IN:GET_LOCATION_HOMETOWN',
 'IN:GET_LOCATION_SCHOOL',
 'IN:GET_LOCATION_WORK',
 'IN:NEGATION',
 'IN:UNINTELLIGIBLE',
 'IN:UNSUPPORTED',
 'IN:UNSUPPORTED_EVENT',
 'IN:UNSUPPORTED_NAVIGATION',
 'IN:UPDATE_DIRECTIONS']

indic_top_slots = ['SL:AMOUNT',
 'SL:ATTENDEE_EVENT',
 'SL:ATTRIBUTE_EVENT',
 'SL:CATEGORY_EVENT',
 'SL:CATEGORY_LOCATION',
 'SL:COMBINE',
 'SL:CONTACT',
 'SL:CONTACT_RELATED',
 'SL:DATE_TIME',
 'SL:DATE_TIME_ARRIVAL',
 'SL:DATE_TIME_DEPARTURE',
 'SL:DESTINATION',
 'SL:GROUP',
 'SL:LOCATION',
 'SL:LOCATION_CURRENT',
 'SL:LOCATION_MODIFIER',
 'SL:LOCATION_USER',
 'SL:LOCATION_WORK',
 'SL:METHOD_TRAVEL',
 'SL:NAME_EVENT',
 'SL:OBSTRUCTION',
 'SL:OBSTRUCTION_AVOID',
 'SL:ORDINAL',
 'SL:ORGANIZER_EVENT',
 'SL:PATH',
 'SL:PATH_AVOID',
 'SL:POINT_ON_MAP',
 'SL:ROAD_CONDITION',
 'SL:ROAD_CONDITION_AVOID',
 'SL:SEARCH_RADIUS',
 'SL:SOURCE',
 'SL:TYPE_RELATION',
 'SL:UNIT_DISTANCE',
 'SL:WAYPOINT',
 'SL:WAYPOINT_ADDED',
 'SL:WAYPOINT_AVOID']


intents_slots = {
    "itop": { 
             "intents" : itop_intents,
             "slots": itop_slots
            },
    "indic-TOP": {
                 "intents": indic_top_intents,
                 "slots": indic_top_slots
                 },
    "indic-atis": {
                    "intents": indic_atis_intents,
                    "slots": indic_atis_slots
                  }
}