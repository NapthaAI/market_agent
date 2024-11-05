import uuid 
import json 
import asyncio 
import re 
import time 
import ast 
import aiohttp 
import logging 
import os 
import tiktoken 
import openai 
import anthropic 
import yaml 
import random 
import string 
from typing import Dict 
from typing import Any 
from typing import List 
from typing import Optional 
from typing import Type 
from typing import Union 
from datetime import datetime 
from pydantic.main import BaseModel 
from pydantic.fields import Field 
from tenacity import retry 
from tenacity.stop import stop_after_attempt 
from tenacity.wait import wait_random_exponential 
from typing import Literal 
from pydantic_core._pydantic_core import ValidationError 
from pydantic.fields import computed_field 
from typing import Tuple 
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam 
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam 
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam 
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam 
from openai.types.chat.chat_completion_function_message_param import ChatCompletionFunctionMessageParam 
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam 
from openai.types.chat.chat_completion import ChatCompletion 
from openai.types.shared_params.response_format_text import ResponseFormatText 
from openai.types.shared_params.response_format_json_object import ResponseFormatJSONObject 
from openai.types.shared_params.response_format_text import ResponseFormatText 
from openai.types.shared_params.response_format_json_object import ResponseFormatJSONObject 
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema 
from anthropic.types.message_param import MessageParam 
from anthropic.types.text_block import TextBlock 
from anthropic.types.tool_param import ToolParam 
from anthropic.types.beta.prompt_caching.prompt_caching_beta_tool_param import PromptCachingBetaToolParam 
from anthropic.types.beta.prompt_caching.prompt_caching_beta_text_block_param import PromptCachingBetaTextBlockParam 
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam 
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam 
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam 
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam 
from openai.types.chat.chat_completion_function_message_param import ChatCompletionFunctionMessageParam 
from anthropic.types.beta.prompt_caching.prompt_caching_beta_cache_control_ephemeral_param import PromptCachingBetaCacheControlEphemeralParam 
from anthropic.types.beta.prompt_caching.prompt_caching_beta_message_param import PromptCachingBetaMessageParam 
from openai.types.shared_params.function_definition import FunctionDefinition 
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema 
from openai.types.shared_params.response_format_json_schema import JSONSchema 
from pydantic.functional_validators import model_validator 
from typing_extensions import Self 
from anthropic.types.tool_use_block import ToolUseBlock 
from anthropic.types.message import Message as AnthropicMessage 
from anthropic.types.beta.prompt_caching.prompt_caching_beta_message import PromptCachingBetaMessage 
from builtins import str 
from openai.types.chat.chat_completion_named_tool_choice_param import ChatCompletionNamedToolChoiceParam 
from openai.types.chat.chat_completion_function_call_option_param import ChatCompletionFunctionCallOptionParam 
from dataclasses import dataclass 
from dataclasses import field 
from dotenv.main import load_dotenv 
from anthropic.types.message_create_params import ToolChoiceToolChoiceTool 
from copy import deepcopy 
from functools import cached_property 
from abc import ABC 
from abc import abstractmethod 
from typing import Generic 
from enum import Enum 
from pydantic_settings.main import BaseSettings 
from anthropic.types.model_param import ModelParam
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam
)
from openai.types.chat.completion_create_params import (
    ResponseFormat,
    FunctionCall
)
from dotenv import load_dotenv
from pathlib import Path
from market_agent.schemas import InputSchema, LLMConfig, OrchestratorConfig, MultiAgentEnvironment, Protocol, GlobalAction, LocalAction, LocalObservation, GlobalObservation, Mechanism, LocalEnvironmentStep, EnvironmentStep, ActionSpace, ObservationSpace
from naptha_sdk.utils import get_logger
from anthropic.types.beta.prompt_caching import message_create_params
from typing import TypeVar

T = TypeVar('T')

logger = get_logger(__name__)

load_dotenv()

class PerceptionSchema(BaseModel):
    monologue: str = Field(..., description="Agent's internal monologue about the perceived market situation")
    strategy: str= Field(..., description="Agent's strategy given the current market situation")

class ReflectionSchema(BaseModel):
    reflection: str = Field(..., description="Reflection on the observation and surplus based on the last action")
    strategy_update: str = Field(..., description="Updated strategy based on the reflection, surplus, and previous strategy")

def msg_dict_to_oai(messages: List[Dict[str, Any]]) -> List[ChatCompletionMessageParam]:
        def convert_message(msg: Dict[str, Any]) -> ChatCompletionMessageParam:
            role = msg["role"]
            if role == "system":
                return ChatCompletionSystemMessageParam(role=role, content=msg["content"])
            elif role == "user":
                return ChatCompletionUserMessageParam(role=role, content=msg["content"])
            elif role == "assistant":
                assistant_msg = ChatCompletionAssistantMessageParam(role=role, content=msg.get("content"))
                if "function_call" in msg:
                    assistant_msg["function_call"] = msg["function_call"]
                if "tool_calls" in msg:
                    assistant_msg["tool_calls"] = msg["tool_calls"]
                return assistant_msg
            elif role == "tool":
                return ChatCompletionToolMessageParam(role=role, content=msg["content"], tool_call_id=msg["tool_call_id"])
            elif role == "function":
                return ChatCompletionFunctionMessageParam(role=role, content=msg["content"], name=msg["name"])
            else:
                raise ValueError(f"Unknown role: {role}")

        return [convert_message(msg) for msg in messages]

def msg_dict_to_anthropic(messages: List[Dict[str, Any]],use_cache:bool=True,use_prefill:bool=False) -> Tuple[List[PromptCachingBetaTextBlockParam],List[MessageParam]]:
        def create_anthropic_system_message(system_message: Optional[Dict[str, Any]],use_cache:bool=True) -> List[PromptCachingBetaTextBlockParam]:
            if system_message and system_message["role"] == "system":
                text = system_message["content"]
                if use_cache:
                    return [PromptCachingBetaTextBlockParam(type="text", text=text, cache_control=PromptCachingBetaCacheControlEphemeralParam(type="ephemeral"))]
                else:
                    return [PromptCachingBetaTextBlockParam(type="text", text=text)]
            return []

        def convert_message(msg: Dict[str, Any],use_cache:bool=False) -> Union[PromptCachingBetaMessageParam, None]:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                return None
            
            if isinstance(content, str):
                if not use_cache:
                    content = [PromptCachingBetaTextBlockParam(type="text", text=content)]
                else:
                    content = [PromptCachingBetaTextBlockParam(type="text", text=content,cache_control=PromptCachingBetaCacheControlEphemeralParam(type='ephemeral'))]
            elif isinstance(content, list):
                if not use_cache:
                    content = [
                        PromptCachingBetaTextBlockParam(type="text", text=block) if isinstance(block, str)
                        else PromptCachingBetaTextBlockParam(type="text", text=block["text"]) for block in content
                    ]
                else:
                    content = [
                        PromptCachingBetaTextBlockParam(type="text", text=block, cache_control=PromptCachingBetaCacheControlEphemeralParam(type='ephemeral')) if isinstance(block, str)
                        else PromptCachingBetaTextBlockParam(type="text", text=block["text"], cache_control=PromptCachingBetaCacheControlEphemeralParam(type='ephemeral')) for block in content
                    ]
            else:
                raise ValueError("Invalid content type")
            
            return PromptCachingBetaMessageParam(role=role, content=content)
        converted_messages = []
        system_message = []
        num_messages = len(messages)
        if use_cache:
            use_cache_ids = set([num_messages - 1, max(0, num_messages - 3)])
        else:
            use_cache_ids = set()
        for i,message in enumerate(messages):
            if message["role"] == "system":
                system_message= create_anthropic_system_message(message,use_cache=use_cache)
            else:
                
                use_cache_final = use_cache if  i in use_cache_ids else False
                converted_messages.append(convert_message(message,use_cache= use_cache_final))

        
        return system_message, [msg for msg in converted_messages if msg is not None]

class StructuredTool(BaseModel):
    """ Supported type by OpenAI Structured Output:
    String, Number, Boolean, Integer, Object, Array, Enum, anyOf
    Root must be Object, not anyOf
    Not supported by OpenAI Structured Output: 
    For strings: minLength, maxLength, pattern, format
    For numbers: minimum, maximum, multipleOf
    For objects: patternProperties, unevaluatedProperties, propertyNames, minProperties, maxProperties
    For arrays: unevaluatedItems, contains, minContains, maxContains, minItems, maxItems, uniqueItems
    oai_reference: https://platform.openai.com/docs/guides/structured-outputs/how-to-use """

    json_schema: Optional[Dict[str, Any]] = None
    schema_name: str = Field(default = "generate_structured_output")
    schema_description: str = Field(default ="Generate a structured output based on the provided JSON schema.")
    instruction_string: str = Field(default = "Please follow this JSON schema for your response:")
    strict_schema: bool = True

    @computed_field
    @property
    def schema_instruction(self) -> str:
        return f"{self.instruction_string}: {self.json_schema}"

    def get_openai_tool(self) -> Optional[ChatCompletionToolParam]:
        if self.json_schema:
            return ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=self.schema_name,
                    description=self.schema_description,
                    parameters=self.json_schema
                )
            )
        return None

    def get_anthropic_tool(self) -> Optional[PromptCachingBetaToolParam]:
        if self.json_schema:
            return PromptCachingBetaToolParam(
                name=self.schema_name,
                description=self.schema_description,
                input_schema=self.json_schema,
                cache_control=PromptCachingBetaCacheControlEphemeralParam(type='ephemeral')
            )
        return None
    def get_openai_json_schema_response(self) -> Optional[ResponseFormatJSONSchema]:

        if self.json_schema:
            schema = JSONSchema(name=self.schema_name,description=self.schema_description,schema=self.json_schema,strict=self.strict_schema)
            return ResponseFormatJSONSchema(type="json_schema", json_schema=schema)
        return None

def parse_json_string(content: str) -> Optional[Dict[str, Any]]:
    # Remove any leading/trailing whitespace and newlines
    cleaned_content = content.strip()
    
    # Remove markdown code block syntax if present
    cleaned_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned_content, flags=re.MULTILINE)
    
    try:
        # First, try to parse as JSON
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        try:
            # If JSON parsing fails, try to evaluate as a Python literal
            return ast.literal_eval(cleaned_content)
        except (SyntaxError, ValueError):
            # If both methods fail, try to find and parse a JSON-like structure
            json_match = re.search(r'(\{[^{}]*\{.*?\}[^{}]*\}|\{.*?\})', cleaned_content, re.DOTALL)
            if json_match:
                try:
                    # Normalize newlines, replace single quotes with double quotes, and unescape quotes
                    json_str = json_match.group(1).replace('\n', '').replace("'", '"').replace('\\"', '"')
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
    
    # If all parsing attempts fail, return None
    return None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None

class GeneratedJsonObject(BaseModel):
    name: str
    object: Dict[str, Any]

class LLMOutput(BaseModel):
    raw_result: Union[str, dict, ChatCompletion, AnthropicMessage, PromptCachingBetaMessage]
    completion_kwargs: Optional[Dict[str, Any]] = None
    start_time: float
    end_time: float
    source_id: str
    client: Optional[Literal["openai", "anthropic","vllm","litellm"]] = Field(default=None)

    @property
    def time_taken(self) -> float:
        return self.end_time - self.start_time

    @computed_field
    @property
    def str_content(self) -> Optional[str]:
        return self._parse_result()[0]

    @computed_field
    @property
    def json_object(self) -> Optional[GeneratedJsonObject]:
        return self._parse_result()[1]
    
    @computed_field
    @property
    def error(self) -> Optional[str]:
        return self._parse_result()[3]

    @computed_field
    @property
    def contains_object(self) -> bool:
        return self._parse_result()[1] is not None
    
    @computed_field
    @property
    def usage(self) -> Optional[Usage]:
        return self._parse_result()[2]

    @computed_field
    @property
    def result_provider(self) -> Optional[Literal["openai", "anthropic","vllm","litellm"]]:
        return self.search_result_provider() if self.client is None else self.client
    
    @model_validator(mode="after")
    def validate_provider_and_client(self) -> Self:
        if self.client is not None and self.result_provider != self.client:
            raise ValueError(f"The inferred result provider '{self.result_provider}' does not match the specified client '{self.client}'")
        return self
    
    
    def search_result_provider(self) -> Optional[Literal["openai", "anthropic"]]:
        try:
            oai_completion = ChatCompletion.model_validate(self.raw_result)
            return "openai"
        except ValidationError:
            try:
                anthropic_completion = AnthropicMessage.model_validate(self.raw_result)
                return "anthropic"
            except ValidationError:
                try:
                    antrhopic_beta_completion = PromptCachingBetaMessage.model_validate(self.raw_result)
                    return "anthropic"
                except ValidationError:
                    return None

    def _parse_json_string(self, content: str) -> Optional[Dict[str, Any]]:
        return parse_json_string(content)
    
    

    def _parse_oai_completion(self,chat_completion:ChatCompletion) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage], None]:
        message = chat_completion.choices[0].message
        content = message.content

        json_object = None
        usage = None

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            name = tool_call.function.name
            try:
                object_dict = json.loads(tool_call.function.arguments)
                json_object = GeneratedJsonObject(name=name, object=object_dict)
            except json.JSONDecodeError:
                json_object = GeneratedJsonObject(name=name, object={"raw": tool_call.function.arguments})
        elif content is not None:
            if self.completion_kwargs:
                name = self.completion_kwargs.get("response_format",{}).get("json_schema",{}).get("name",None)
            else:
                name = None
            parsed_json = self._parse_json_string(content)
            if parsed_json:
                
                json_object = GeneratedJsonObject(name="parsed_content" if name is None else name,
                                                   object=parsed_json)
                content = None  # Set content to None when we have a parsed JSON object
                #print(f"parsed_json: {parsed_json} with name")
        if chat_completion.usage:
            usage = Usage(
                prompt_tokens=chat_completion.usage.prompt_tokens,
                completion_tokens=chat_completion.usage.completion_tokens,
                total_tokens=chat_completion.usage.total_tokens
            )

        return content, json_object, usage, None

    def _parse_anthropic_message(self, message: Union[AnthropicMessage, PromptCachingBetaMessage]) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage],None]:
        content = None
        json_object = None
        usage = None

        if message.content:
            first_content = message.content[0]
            if isinstance(first_content, TextBlock):
                content = first_content.text
                parsed_json = self._parse_json_string(content)
                if parsed_json:
                    json_object = GeneratedJsonObject(name="parsed_content", object=parsed_json)
                    content = None  # Set content to None when we have a parsed JSON object
            elif isinstance(first_content, ToolUseBlock):
                name = first_content.name
                input_dict : Dict[str,Any] = first_content.input # type: ignore  # had to ignore due to .input being of object class
                json_object = GeneratedJsonObject(name=name, object=input_dict)

        if hasattr(message, 'usage'):
            usage = Usage(
                prompt_tokens=message.usage.input_tokens,
                completion_tokens=message.usage.output_tokens,
                total_tokens=message.usage.input_tokens + message.usage.output_tokens,
                cache_creation_input_tokens=getattr(message.usage, 'cache_creation_input_tokens', None),
                cache_read_input_tokens=getattr(message.usage, 'cache_read_input_tokens', None)
            )

        return content, json_object, usage, None
    

    def _parse_result(self) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage],Optional[str]]:
        provider = self.result_provider
        if getattr(self.raw_result, "error", None):
            return None, None, None,  getattr(self.raw_result, "error", None)
        if provider == "openai":
            return self._parse_oai_completion(ChatCompletion.model_validate(self.raw_result))
        elif provider == "anthropic":
            try: #beta first
                return self._parse_anthropic_message(PromptCachingBetaMessage.model_validate(self.raw_result))
            except ValidationError:
                return self._parse_anthropic_message(AnthropicMessage.model_validate(self.raw_result))
        elif provider == "vllm":
             return self._parse_oai_completion(ChatCompletion.model_validate(self.raw_result))
        elif provider == "litellm":
            return self._parse_oai_completion(ChatCompletion.model_validate(self.raw_result))
        else:
            raise ValueError(f"Unsupported result provider: {provider}")

    class Config:
        arbitrary_types_allowed = True

class LLMPromptContext(BaseModel):
    id: str
    system_string: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    new_message: str
    prefill: str = Field(default="Here's the valid JSON object response:```json", description="prefill assistant response with an instruction")
    postfill: str = Field(default="\n\nPlease provide your response in JSON format.", description="postfill user response with an instruction")
    structured_output : Optional[StructuredTool] = None
    use_schema_instruction: bool = Field(default=False, description="Whether to use the schema instruction")
    llm_config: LLMConfig
    use_history: bool = Field(default=True, description="Whether to use the history")
    
    @computed_field
    @property
    def oai_response_format(self) -> Optional[ResponseFormat]:
        if self.llm_config.response_format == "text":
            return ResponseFormatText(type="text")
        elif self.llm_config.response_format == "json_object":
            return ResponseFormatJSONObject(type="json_object")
        elif self.llm_config.response_format == "structured_output":
            assert self.structured_output is not None, "Structured output is not set"
            return self.structured_output.get_openai_json_schema_response()
        else:
            return None


    @computed_field
    @property
    def use_prefill(self) -> bool:
        if self.llm_config.client in ['anthropic','vllm','litellm'] and  self.llm_config.response_format in ["json_beg"]:

            return True
        else:
            return False
        
    @computed_field
    @property
    def use_postfill(self) -> bool:
        if self.llm_config.client == 'openai' and 'json' in self.llm_config.response_format and not self.use_schema_instruction:
            return True

        else:
            return False
        
    @computed_field
    @property
    def system_message(self) -> Optional[Dict[str, str]]:
        content= self.system_string if self.system_string  else ""
        if self.use_schema_instruction and self.structured_output:
            content = "\n".join([content,self.structured_output.schema_instruction])
        return {"role":"system","content":content} if len(content)>0 else None
    
    @computed_field
    @property
    def messages(self)-> List[Dict[str, Any]]:
        messages = [self.system_message] if self.system_message is not None else []
        if  self.use_history and self.history:
            messages+=self.history
        messages.append({"role":"user","content":self.new_message})
        if self.use_prefill:
            prefill_message = {"role":"assistant","content":self.prefill}
            messages.append(prefill_message)
        elif self.use_postfill:
            messages[-1]["content"] = messages[-1]["content"] + self.postfill
        return messages
    
    @computed_field
    @property
    def oai_messages(self)-> List[ChatCompletionMessageParam]:
        return msg_dict_to_oai(self.messages)
    
    @computed_field
    @property
    def anthropic_messages(self) -> Tuple[List[PromptCachingBetaTextBlockParam],List[MessageParam]]:
        return msg_dict_to_anthropic(self.messages, use_cache=self.llm_config.use_cache)
    
    @computed_field
    @property
    def vllm_messages(self) -> List[ChatCompletionMessageParam]:
        return msg_dict_to_oai(self.messages)
        
    def update_llm_config(self,llm_config:LLMConfig) -> 'LLMPromptContext':
        
        return self.model_copy(update={"llm_config":llm_config})
       
    

        
    def add_chat_turn_history(self, llm_output:'LLMOutput'):
        """ add a chat turn to the history without safely model copy just normal append """
        if llm_output.source_id != self.id:
            raise ValueError(f"LLMOutput source_id {llm_output.source_id} does not match the prompt context id {self.id}")
        if self.history is None:
            self.history = []
        self.history.append({"role": "user", "content": self.new_message})
        self.history.append({"role": "assistant", "content": llm_output.str_content or json.dumps(llm_output.json_object.object) if llm_output.json_object else "{}"})
    
    def get_tool(self) -> Union[ChatCompletionToolParam, PromptCachingBetaToolParam, None]:
        if not self.structured_output:
            return None
        if self.llm_config.client in ["openai","vllm","litellm"]:
            return self.structured_output.get_openai_tool()
        elif self.llm_config.client == "anthropic":
            return self.structured_output.get_anthropic_tool()
        else:
            return None

class AnthropicRequest(BaseModel):
    max_tokens: int
    messages: List[PromptCachingBetaMessageParam]
    model: ModelParam
    metadata: message_create_params.Metadata | None = Field(default=None)
    stop_sequences: List[str] | None = Field(default=None)
    stream: bool | None = Field(default=None)
    system: Union[str, List[PromptCachingBetaTextBlockParam]] | None = Field(default=None)
    temperature: float | None = Field(default=None)
    tool_choice: message_create_params.ToolChoice | None = Field(default=None)
    tools: List[PromptCachingBetaToolParam] | None = Field(default=None)
    top_k: int | None = Field(default=None)
    top_p: float | None = Field(default=None)

class OpenAIRequest(BaseModel):
    messages: List[ChatCompletionMessageParam]
    model: str
    frequency_penalty: Optional[float] = Field(default=None)
    function_call: Optional[FunctionCall] = Field(default=None)
    functions: Optional[List[FunctionDefinition]] = Field(default=None)
    logit_bias: Optional[Dict[str, int]] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    n: Optional[int] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=None)
    response_format: Optional[ResponseFormat] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    stop: Optional[Union[str, List[str]]] = Field(default=None)
    stream: Optional[bool] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = Field(default=None)
    tools: Optional[List[ChatCompletionToolParam]] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    user: Optional[str] = Field(default=None)

    class Config:
        extra = 'forbid'

class VLLMConfig(BaseModel):
    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=
        ("If true, the generation prompt will be added to the chat template. "
         "This is a parameter used by chat template in tokenizer config of the "
         "model."),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    documents: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=
        ("A list of dicts representing documents that will be accessible to "
         "the model if it is performing RAG (retrieval-augmented generation)."
         " If the template does not support RAG, this argument will have no "
         "effect. We recommend that each document should be a dict containing "
         "\"title\" and \"text\" keys."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."),
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description=("If specified, the output will follow the JSON schema."),
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be either "
            "'outlines' / 'lm-format-enforcer'"))
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."))

class VLLMRequest(OpenAIRequest,VLLMConfig):
    pass

class OAIApiFromFileConfig(BaseModel):
 requests_filepath: str
 save_filepath: str
 api_key: str
 request_url:str =  Field("https://api.openai.com/v1/embeddings",description="The url to use for generating embeddings")
 max_requests_per_minute: float = Field(100,description="The maximum number of requests per minute")
 max_tokens_per_minute: float = Field(1_000_000,description="The maximum number of tokens per minute")
 max_attempts:int = Field(5,description="The maximum number of attempts to make for each request")
 logging_level:int = Field(20,description="The logging level to use for the request")
 token_encoding_name: str = Field("cl100k_base",description="The token encoding scheme to use for calculating request sizes")

@dataclass
class StatusTracker:
    """
    A data class that tracks the progress and status of API request processing.
    
    This class is designed to hold counters for various outcomes of API requests
    (such as successes, failures, and specific types of errors) and other relevant
    metadata to manage and monitor the execution flow of the script.
    
    Attributes:
    - num_tasks_started: The total number of tasks that have been started.
    - num_tasks_in_progress: The current number of tasks that are in progress.
      The script continues running as long as this number is greater than 0.
    - num_tasks_succeeded: The total number of tasks that have completed successfully.
    - num_tasks_failed: The total number of tasks that have failed.
    - num_rate_limit_errors: The count of errors received due to hitting the API's rate limits.
    - num_api_errors: The count of API-related errors excluding rate limit errors.
    - num_other_errors: The count of errors that are neither API errors nor rate limit errors.
    - time_of_last_rate_limit_error: A timestamp (as an integer) of the last time a rate limit error was encountered,
      used to implement a cooling-off period before making subsequent requests.
    
    The class is initialized with all counters set to 0, and the `time_of_last_rate_limit_error`
    set to 0 indicating no rate limit errors have occurred yet.
    """

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = 0  # used to cool off after hitting rate limits

def append_to_jsonl(data, filename: str) -> None:
    """
    Appends a given JSON payload to the end of a JSON Lines (.jsonl) file.

    Parameters:
    - data: The JSON-serializable Python object (e.g., dict, list) to be appended.
    - filename (str): The path to the .jsonl file to which the data will be appended.

    The function converts `data` into a JSON string and appends it to the specified file,
    ensuring that each entry is on a new line, consistent with the JSON Lines format.

    Note:
    - If the specified file does not exist, it will be created.
    - This function does not return any value.
    """
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")

@dataclass
class APIRequest:
    """
    Represents an individual API request with associated metadata and the capability to asynchronously call an API.
    
    Attributes:
    - task_id (int): A unique identifier for the task.
    - request_json (dict): The JSON payload to be sent with the request.
    - token_consumption (int): Estimated number of tokens consumed by the request, used for rate limiting.
    - attempts_left (int): The number of retries left if the request fails.
    - metadata (dict): Additional metadata associated with the request.
    - result (list): A list to store the results or errors from the API call.
    
    This class encapsulates the data and actions related to making an API request, including
    retry logic and error handling.
    """

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """
        Asynchronously sends the API request using aiohttp, handles errors, and manages retries.
        
        Parameters:
        - session (aiohttp.ClientSession): The session object used for HTTP requests.
        - request_url (str): The URL to which the request is sent.
        - request_header (dict): Headers for the request, including authorization.
        - retry_queue (asyncio.Queue): A queue for requests that need to be retried.
        - save_filepath (str): The file path where results or errors should be logged.
        - status_tracker (StatusTracker): A shared object for tracking the status of all API requests.
        
        This method attempts to post the request to the given URL. If the request encounters an error,
        it determines whether to retry based on the remaining attempts and updates the status tracker
        accordingly. Successful requests or final failures are logged to the specified file.
        """
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e

        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                self.metadata["end_time"] = time.time()
                self.metadata["total_time"] = self.metadata["end_time"] - self.metadata["start_time"]
                data = [self.metadata, self.request_json, {"error": str(error)}]
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            self.metadata["end_time"] = time.time()
            self.metadata["total_time"] = self.metadata["end_time"] - self.metadata["start_time"]
            data = [self.metadata, self.request_json, response]
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")

def api_endpoint_from_url(request_url: str) -> str:
    """
    Extracts the API endpoint from a given request URL.

    This function applies a regular expression search to find the API endpoint pattern within the provided URL.
    It supports extracting endpoints from standard OpenAI API URLs, custom Azure OpenAI deployment URLs,
    and vLLM endpoints.

    Parameters:
    - request_url (str): The full URL of the API request.

    Returns:
    - str: The extracted endpoint from the URL. If the URL does not match expected patterns,
      this function may raise an IndexError for accessing a non-existing match group.

    Example:
    - Input: "https://api.openai.com/v1/completions"
      Output: "completions"
    - Input: "https://custom.azurewebsites.net/openai/deployments/my-model/completions"
      Output: "completions"
    - Input: "http://localhost:8000/v1/completions"
      Output: "completions"
    """
    match = re.search("^https://[^/]+/v\\d+/(.+)$", request_url)
    if match is None:
        # for Azure OpenAI deployment urls
        match = re.search(r"^https://[^/]+/openai/deployments/[^/]+/(.+?)(\?|$)", request_url)
        if match is None:
            # for vLLM endpoints
            match = re.search(r"^http://localhost:8000/v\d+/(.+)$", request_url)
            if match is None:
                raise ValueError(f"Invalid URL: {request_url}")
    return match[1]

def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Supports completion, embedding, and Anthropic message requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    'Expecting either string or list of strings for "prompt" field in completion request'
                )
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    elif api_endpoint == "messages":  # Anthropic API
        messages = request_json.get("messages", [])
        num_tokens = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                num_tokens += len(encoding.encode(content))
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        num_tokens += len(encoding.encode(item["text"]))
        
        max_tokens = request_json.get("max_tokens", 0)
        num_tokens += max_tokens  # Add the max_tokens to account for the response
        return num_tokens
    
    else:
        raise NotImplementedError(
            f'API endpoint "{api_endpoint}" not implemented in this script'
        )

def task_id_generator_function():
    """
    Generates a sequence of integer task IDs, starting from 0 and incrementing by 1 each time.

    Yields:
    - An integer representing the next task ID in the sequence.

    This generator function is useful for assigning unique identifiers to tasks or requests
    in a sequence, ensuring each has a distinct ID for tracking and reference purposes.
    """
    task_id = 0
    while True:
        yield task_id
        task_id += 1

async def process_api_requests_from_file(
        api_cfg: OAIApiFromFileConfig
):
    """
    Asynchronously processes API requests from a given file, executing them in parallel
    while adhering to specified rate limits for requests and tokens per minute.
    
    This function reads a file containing JSONL-formatted API requests, sends these requests
    concurrently to the specified API endpoint, and handles retries for failed attempts,
    all the while ensuring that the execution does not exceed the given rate limits.
    
    Parameters:
    - requests_filepath: Path to the file containing the JSONL-formatted API requests.
    - save_filepath: Path to the file where results or logs should be saved.
    - request_url: The API endpoint URL to which the requests will be sent.
    - api_key: The API key for authenticating requests to the endpoint.
    - max_requests_per_minute: The maximum number of requests allowed per minute.
    - max_tokens_per_minute: The maximum number of tokens (for rate-limited APIs) that can be used per minute.
    - token_encoding_name: Name of the token encoding scheme used for calculating request sizes.
    - max_attempts: The maximum number of attempts for each request in case of failures.
    - logging_level: The logging level to use for reporting the process's progress and issues.
    
    The function initializes necessary tracking structures, sets up asynchronous HTTP sessions,
    and manages request retries and rate limiting. It logs the progress and any issues encountered
    during the process to facilitate monitoring and debugging.
    """
    #extract variables from config
    requests_filepath = api_cfg.requests_filepath
    save_filepath = api_cfg.save_filepath
    request_url = api_cfg.request_url
    api_key = api_cfg.api_key
    max_requests_per_minute = api_cfg.max_requests_per_minute
    max_tokens_per_minute = api_cfg.max_tokens_per_minute
    token_encoding_name = api_cfg.token_encoding_name
    max_attempts = api_cfg.max_attempts
    logging_level = api_cfg.logging_level
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}
    # use api-key header for Azure deployments
    if '/deployments' in request_url:
        request_header = {"api-key": f"{api_key}"}
    # Add Anthropic-specific headers
    if 'anthropic.com' in request_url:
        request_header = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "anthropic-beta": "prompt-caching-2024-07-31"
        }

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 1, 2, 3, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")
        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))
                            metadata, actual_request = request_json  # Unpack the list
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=actual_request,
                                token_consumption=num_tokens_consumed_from_request(
                                    actual_request, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=metadata,
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warn(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # after finishing, log final status
        logging.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )

class RequestLimits(BaseModel):
    max_requests_per_minute: int = Field(default=50,description="The maximum number of requests per minute for the API")
    max_tokens_per_minute: int = Field(default=100000,description="The maximum number of tokens per minute for the API")
    provider: Literal["openai", "anthropic", "vllm", "litellm"] = Field(default="openai",description="The provider of the API")

class ParallelAIUtilities:
    def __init__(self, oai_request_limits: Optional[RequestLimits] = None, 
                 anthropic_request_limits: Optional[RequestLimits] = None, 
                 vllm_request_limits: Optional[RequestLimits] = None,
                 litellm_request_limits: Optional[RequestLimits] = None,
                 local_cache: bool = True,
                 cache_folder: Optional[str] = None):
        load_dotenv()
        self.openai_key = os.getenv("OPENAI_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.vllm_key = os.getenv("VLLM_API_KEY")
        self.vllm_endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
        self.litellm_endpoint = os.getenv("LITELLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
        self.litellm_key = os.getenv("LITELLM_API_KEY")
        self.oai_request_limits = oai_request_limits if oai_request_limits else RequestLimits(max_requests_per_minute=500,max_tokens_per_minute=200000,provider="openai")
        self.anthropic_request_limits = anthropic_request_limits if anthropic_request_limits else RequestLimits(max_requests_per_minute=50,max_tokens_per_minute=40000,provider="anthropic")
        self.vllm_request_limits = vllm_request_limits if vllm_request_limits else RequestLimits(max_requests_per_minute=500,max_tokens_per_minute=200000,provider="vllm")
        self.litellm_request_limits = litellm_request_limits if litellm_request_limits else RequestLimits(max_requests_per_minute=500,max_tokens_per_minute=200000,provider="litellm")
        self.local_cache = local_cache
        self.cache_folder = self._setup_cache_folder(cache_folder)

    def _setup_cache_folder(self, cache_folder: Optional[str]) -> str:
        if cache_folder:
            full_path = os.path.abspath(cache_folder)
        else:
            # Go up two levels from the current file's directory to reach the project root
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            full_path = os.path.join(repo_root, 'outputs', 'inference_cache')
        
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def _create_prompt_hashmap(self, prompts: List[LLMPromptContext]) -> Dict[str, LLMPromptContext]:
        return {p.id: p for p in prompts}
    
    def _update_prompt_history(self, prompts: List[LLMPromptContext], llm_outputs: List[LLMOutput]):
        prompt_hashmap = self._create_prompt_hashmap(prompts)
        for output in llm_outputs:
            prompt_hashmap[output.source_id].add_chat_turn_history(output)
        return list(prompt_hashmap.values())

    async def run_parallel_ai_completion(self, prompts: List[LLMPromptContext], update_history:bool=True) -> List[LLMOutput]:
        openai_prompts = [p for p in prompts if p.llm_config.client == "openai"]
        anthropic_prompts = [p for p in prompts if p.llm_config.client == "anthropic"]
        vllm_prompts = [p for p in prompts if p.llm_config.client == "vllm"] 
        litellm_prompts = [p for p in prompts if p.llm_config.client == "litellm"]
        tasks = []
        if openai_prompts:
            tasks.append(self._run_openai_completion(openai_prompts))
        if anthropic_prompts:
            tasks.append(self._run_anthropic_completion(anthropic_prompts))
        if vllm_prompts:
            tasks.append(self._run_vllm_completion(vllm_prompts))
        if litellm_prompts:
            tasks.append(self._run_litellm_completion(litellm_prompts))

        results = await asyncio.gather(*tasks)
        flattened_results = [item for sublist in results for item in sublist]
        if update_history:
            prompts = self._update_prompt_history(prompts, flattened_results)
        
        # Flatten the results list
        return flattened_results

    async def _run_openai_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_file = os.path.join(self.cache_folder, f'openai_requests_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'openai_results_{timestamp}.jsonl')
        self._prepare_requests_file(prompts, "openai", requests_file)
        config = self._create_oai_completion_config(prompts[0], requests_file, results_file)
        if config:
            try:
                await process_api_requests_from_file(config)
                return self._parse_results_file(results_file,client="openai")
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    async def _run_anthropic_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_file = os.path.join(self.cache_folder, f'anthropic_requests_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'anthropic_results_{timestamp}.jsonl')
        self._prepare_requests_file(prompts, "anthropic", requests_file)
        config = self._create_anthropic_completion_config(prompts[0], requests_file, results_file)
        if config:
            try:
                await process_api_requests_from_file(config)
                return self._parse_results_file(results_file,client="anthropic")
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []
    
    async def _run_vllm_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_file = os.path.join(self.cache_folder, f'vllm_requests_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'vllm_results_{timestamp}.jsonl')
        self._prepare_requests_file(prompts, "vllm", requests_file)
        config = self._create_vllm_completion_config(prompts[0], requests_file, results_file)
        if config:
            try:
                await process_api_requests_from_file(config)
                return self._parse_results_file(results_file,client="vllm")
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []
    
    async def _run_litellm_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_file = os.path.join(self.cache_folder, f'litellm_requests_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'litellm_results_{timestamp}.jsonl')
        self._prepare_requests_file(prompts, "litellm", requests_file)
        config = self._create_litellm_completion_config(prompts[0], requests_file, results_file)
        if config:
            try:
                await process_api_requests_from_file(config)
                return self._parse_results_file(results_file,client="litellm")
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    

    def _prepare_requests_file(self, prompts: List[LLMPromptContext], client: str, filename: str):
        requests = []
        for prompt in prompts:
            request = self._convert_prompt_to_request(prompt, client)
            if request:
                metadata = {
                    "prompt_context_id": prompt.id,
                    "start_time": time.time(),
                    "end_time": None,
                    "total_time": None
                }
                requests.append([metadata, request])
        
        with open(filename, 'w') as f:
            for request in requests:
                json.dump(request, f)
                f.write('\n')

    def _validate_anthropic_request(self, request: Dict[str, Any]) -> bool:
        try:
            anthropic_request = AnthropicRequest(**request)
            return True
        except Exception as e:
            raise ValidationError(f"Error validating Anthropic request: {e} with request: {request}")
    
    def _validate_openai_request(self, request: Dict[str, Any]) -> bool:
        try:
            openai_request = OpenAIRequest(**request)
            return True
        except Exception as e:
            raise ValidationError(f"Error validating OpenAI request: {e} with request: {request}")
        
    def _validate_vllm_request(self, request: Dict[str, Any]) -> bool:
        try:
            vllm_request = VLLMRequest(**request)
            return True
        except Exception as e:
            # Instead of raising ValidationError, we'll return False
            raise ValidationError(f"Error validating VLLM request: {e} with request: {request}")
        

    
    def _get_openai_request(self, prompt: LLMPromptContext) -> Optional[Dict[str, Any]]:
        messages = prompt.oai_messages
        request = {
            "model": prompt.llm_config.model,
            "messages": messages,
            "max_tokens": prompt.llm_config.max_tokens,
            "temperature": prompt.llm_config.temperature,
        }
        if prompt.oai_response_format:
            request["response_format"] = prompt.oai_response_format
        if prompt.llm_config.response_format == "tool" and prompt.structured_output:
            tool = prompt.get_tool()
            if tool:
                request["tools"] = [tool]
                request["tool_choice"] = {"type": "function", "function": {"name": prompt.structured_output.schema_name}}
        if self._validate_openai_request(request):
            return request
        else:
            return None
    
    def _get_anthropic_request(self, prompt: LLMPromptContext) -> Optional[Dict[str, Any]]:
        system_content, messages = prompt.anthropic_messages    
        request = {
            "model": prompt.llm_config.model,
            "max_tokens": prompt.llm_config.max_tokens,
            "temperature": prompt.llm_config.temperature,
            "messages": messages,
            "system": system_content if system_content else None
        }
        if prompt.llm_config.response_format == "tool" and prompt.structured_output:
            tool = prompt.get_tool()
            if tool:
                request["tools"] = [tool]
                request["tool_choice"] = ToolChoiceToolChoiceTool(name=prompt.structured_output.schema_name, type="tool")

        if self._validate_anthropic_request(request):
            return request
        else:
            return None
        
    def _get_vllm_request(self, prompt: LLMPromptContext) -> Optional[Dict[str, Any]]:
        messages = prompt.vllm_messages  # Use vllm_messages instead of oai_messages
        request = {
            "model": prompt.llm_config.model,
            "messages": messages,
            "max_tokens": prompt.llm_config.max_tokens,
            "temperature": prompt.llm_config.temperature,
        }
        if prompt.llm_config.response_format == "tool" and prompt.structured_output:
            tool = prompt.get_tool()
            if tool:
                request["tools"] = [tool]
                request["tool_choice"] = {"type": "function", "function": {"name": prompt.structured_output.schema_name}}
        if prompt.llm_config.response_format == "json_object":
            raise ValueError("VLLM does not support json_object response format otherwise infinite whitespaces are returned")
        if prompt.oai_response_format and prompt.oai_response_format:
            request["response_format"] = prompt.oai_response_format
        
        if self._validate_vllm_request(request):
            return request
        else:
            return None
        
    def _get_litellm_request(self, prompt: LLMPromptContext) -> Optional[Dict[str, Any]]:
        if prompt.llm_config.response_format == "json_object":
            raise ValueError("VLLM does not support json_object response format otherwise infinite whitespaces are returned")
        return self._get_openai_request(prompt)
        
    def _convert_prompt_to_request(self, prompt: LLMPromptContext, client: str) -> Optional[Dict[str, Any]]:
        if client == "openai":
            return self._get_openai_request(prompt)
        elif client == "anthropic":
            return self._get_anthropic_request(prompt)
        elif client == "vllm":
            return self._get_vllm_request(prompt)
        elif client =="litellm":
            return self._get_litellm_request(prompt)
        else:
            raise ValueError(f"Invalid client: {client}")


    def _create_oai_completion_config(self, prompt: LLMPromptContext, requests_file: str, results_file: str) -> Optional[OAIApiFromFileConfig]:
        if prompt.llm_config.client == "openai" and self.openai_key:
            return OAIApiFromFileConfig(
                requests_filepath=requests_file,
                save_filepath=results_file,
                request_url="https://api.openai.com/v1/chat/completions",
                api_key=self.openai_key,
                max_requests_per_minute=self.oai_request_limits.max_requests_per_minute,
                max_tokens_per_minute=self.oai_request_limits.max_tokens_per_minute,
                token_encoding_name="cl100k_base",
                max_attempts=5,
                logging_level=20,
            )
        return None

    def _create_anthropic_completion_config(self, prompt: LLMPromptContext, requests_file: str, results_file: str) -> Optional[OAIApiFromFileConfig]:
        if prompt.llm_config.client == "anthropic" and self.anthropic_key:
            return OAIApiFromFileConfig(
                requests_filepath=requests_file,
                save_filepath=results_file,
                request_url="https://api.anthropic.com/v1/messages",
                api_key=self.anthropic_key,
                max_requests_per_minute=self.anthropic_request_limits.max_requests_per_minute,
                max_tokens_per_minute=self.anthropic_request_limits.max_tokens_per_minute,
                token_encoding_name="cl100k_base",
                max_attempts=5,
                logging_level=20,
            )
        return None
    
    def _create_vllm_completion_config(self, prompt: LLMPromptContext, requests_file: str, results_file: str) -> Optional[OAIApiFromFileConfig]:
        if prompt.llm_config.client == "vllm":
            return OAIApiFromFileConfig(
                requests_filepath=requests_file,
                save_filepath=results_file,
                request_url=self.vllm_endpoint,
                api_key=self.vllm_key if self.vllm_key else "",
                max_requests_per_minute=self.vllm_request_limits.max_requests_per_minute,
                max_tokens_per_minute=self.vllm_request_limits.max_tokens_per_minute,
                token_encoding_name="cl100k_base",
                max_attempts=5,
                logging_level=20,
            )
        return None
    
    def _create_litellm_completion_config(self, prompt: LLMPromptContext, requests_file: str, results_file: str) -> Optional[OAIApiFromFileConfig]:
        if prompt.llm_config.client == "litellm":
            return OAIApiFromFileConfig(
                requests_filepath=requests_file,
                save_filepath=results_file,
                request_url=self.litellm_endpoint,
                api_key=self.litellm_key if self.litellm_key else "",
                max_requests_per_minute=self.litellm_request_limits.max_requests_per_minute,
                max_tokens_per_minute=self.litellm_request_limits.max_tokens_per_minute,
                token_encoding_name="cl100k_base",
                max_attempts=5,
                logging_level=20,
            )
        return None
    

    def _parse_results_file(self, filepath: str,client: Literal["openai", "anthropic", "vllm", "litellm"]) -> List[LLMOutput]:
        results = []
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    llm_output = self._convert_result_to_llm_output(result,client)
                    results.append(llm_output)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {line}")
                    results.append(LLMOutput(raw_result={"error": "JSON decode error"}, completion_kwargs={}, start_time=time.time(), end_time=time.time(), source_id="error"))
                except Exception as e:
                    print(f"Error processing result: {e}")
                    results.append(LLMOutput(raw_result={"error": str(e)}, completion_kwargs={}, start_time=time.time(), end_time=time.time(), source_id="error"))
        return results

    def _convert_result_to_llm_output(self, result: List[Dict[str, Any]],client: Literal["openai", "anthropic", "vllm", "litellm"]) -> LLMOutput:
        metadata, request_data, response_data = result
        
        return LLMOutput(
            raw_result=response_data,
            completion_kwargs=request_data,
            start_time=metadata["start_time"],
            end_time=metadata["end_time"] or time.time(),
            source_id=metadata["prompt_context_id"],
            client=client
        )

    def _delete_files(self, *files):
        for file in files:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error deleting file {file}: {e}")

class TaskPromptSchema(BaseModel):
    """Schema for task prompts."""
    Tasks: str
    Output_schema: Optional[str] = None
    Assistant: Optional[str] = None

class PromptTemplateVariables(BaseModel):
    """Schema for prompt template variables."""
    role: str
    persona: Optional[str] = None
    objectives: Optional[str] = None
    task: Union[str, List[str]]
    datetime: str
    doc_list: str
    pydantic_schema: Optional[str] = None
    output_format: str

def extract_json_from_response(response_string):
    try:
        # Extract JSON data from the response string
        start_index = response_string.find('{')
        end_index = response_string.rfind('}') + 1
        
        if start_index == -1 or end_index == 0:
            return None
        
        json_data = response_string[start_index:end_index]
        
        # Attempt to parse the extracted JSON data
        try:
            return json.loads(json_data)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(json_data)
            except (ValueError, SyntaxError):
                return None
    except Exception as e:
        logger.error(f"Error extracting JSON: {e}")
        return None

class MarketAction(BaseModel):
    price: float = Field(..., description="Price of the order")
    quantity: int = Field(default=1, ge=1,le=1, description="Quantity of the order")

class Bid(MarketAction):
    @computed_field
    @property
    def is_buyer(self) -> bool:
        return True

class Ask(MarketAction):
    @computed_field
    @property
    def is_buyer(self) -> bool:
        return False

class Trade(BaseModel):
    trade_id: int = Field(..., description="Unique identifier for the trade")
    buyer_id: str = Field(..., description="ID of the buyer")
    seller_id: str = Field(..., description="ID of the seller")
    price: float = Field(..., description="The price at which the trade was executed")
    ask_price: float = Field(ge=0, description="The price at which the ask was executed")
    bid_price: float = Field(ge=0, description="The price at which the bid was executed")
    quantity: int = Field(default=1, description="The quantity traded")
    good_name: str = Field(default="consumption_good", description="The name of the good traded")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the trade")

    @model_validator(mode='after')
    def rational_trade(self):
        if self.ask_price > self.bid_price:
            raise ValueError(f"Ask price {self.ask_price} is more than bid price {self.bid_price}")
        return self

class Good(BaseModel):
    name: str
    quantity: float

class Basket(BaseModel):
    cash: float
    goods: List[Good]

    @computed_field
    @cached_property
    def goods_dict(self) -> Dict[str, int]:
        return {good.name: int(good.quantity) for good in self.goods}

    def update_good(self, name: str, quantity: float):
        for good in self.goods:
            if good.name == name:
                good.quantity = quantity
                return
        self.goods.append(Good(name=name, quantity=quantity))

    def get_good_quantity(self, name: str) -> int:
        return int(next((good.quantity for good in self.goods if good.name == name), 0))

class Endowment(BaseModel):
    initial_basket: Basket
    trades: List[Trade] = Field(default_factory=list)
    agent_id: str

    @computed_field
    @property
    def current_basket(self) -> Basket:
        temp_basket = deepcopy(self.initial_basket)

        for trade in self.trades:
            if trade.buyer_id == self.agent_id:
                temp_basket.cash -= trade.price * trade.quantity
                temp_basket.update_good(trade.good_name, temp_basket.get_good_quantity(trade.good_name) + trade.quantity)
            elif trade.seller_id == self.agent_id:
                temp_basket.cash += trade.price * trade.quantity
                temp_basket.update_good(trade.good_name, temp_basket.get_good_quantity(trade.good_name) - trade.quantity)

        # Create a new Basket instance with the calculated values
        return Basket(
            cash=temp_basket.cash,
            goods=[Good(name=good.name, quantity=good.quantity) for good in temp_basket.goods]
        )

    def add_trade(self, trade: Trade):
        self.trades.append(trade)
        # Clear the cached property to ensure it's recalculated
        if 'current_basket' in self.__dict__:
            del self.__dict__['current_basket']

    def simulate_trade(self, trade: Trade) -> Basket:
        temp_basket = deepcopy(self.current_basket)

        if trade.buyer_id == self.agent_id:
            temp_basket.cash -= trade.price * trade.quantity
            temp_basket.update_good(trade.good_name, temp_basket.get_good_quantity(trade.good_name) + trade.quantity)
        elif trade.seller_id == self.agent_id:
            temp_basket.cash += trade.price * trade.quantity
            temp_basket.update_good(trade.good_name, temp_basket.get_good_quantity(trade.good_name) - trade.quantity)

        return temp_basket

class PreferenceSchedule(BaseModel):
    num_units: int = Field(..., description="Number of units")
    base_value: float = Field(..., description="Base value for the first unit")
    noise_factor: float = Field(default=0.1, description="Noise factor for value generation")
    is_buyer: bool = Field(default=True, description="Whether the agent is a buyer")

    @computed_field
    @cached_property
    def values(self) -> Dict[int, float]:
        raise NotImplementedError("Subclasses must implement this method")

    @computed_field
    @cached_property
    def initial_endowment(self) -> float:
        raise NotImplementedError("Subclasses must implement this method")

    def get_value(self, quantity: int) -> float:
        return self.values.get(quantity, 0.0)

    def plot_schedule(self, block=False):
        quantities = list(self.values.keys())
        values = list(self.values.values())
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(quantities, values, marker='o')
        plt.title(f"{'Demand' if self.is_buyer else 'Supply'} Schedule")
        plt.xlabel("Quantity")
        plt.ylabel("Value/Cost")
        plt.grid(True)
        plt.show(block=block)

class BuyerPreferenceSchedule(PreferenceSchedule):
    endowment_factor: float = Field(default=1.2, description="Factor to calculate initial endowment")
    is_buyer: bool = Field(default=True, description="Whether the agent is a buyer")

    @computed_field
    @cached_property
    def values(self) -> Dict[int, float]:
        values = {}
        current_value = self.base_value
        for i in range(1, self.num_units + 1):
            # Decrease current_value by 2% to 5% plus noise
            decrement = current_value * random.uniform(0.02, self.noise_factor)
            
            new_value = current_value-decrement
            values[i] = new_value
            current_value = new_value
        return values

    @computed_field
    @cached_property
    def initial_endowment(self) -> float:
        return sum(self.values.values()) * self.endowment_factor

class SellerPreferenceSchedule(PreferenceSchedule):
    is_buyer: bool = Field(default=False, description="Whether the agent is a buyer")
    @computed_field
    @cached_property
    def values(self) -> Dict[int, float]:
        values = {}
        current_value = self.base_value
        for i in range(1, self.num_units + 1):
            # Increase current_value by 2% to 5% plus noise
            increment = current_value * random.uniform(0.02, self.noise_factor)
            new_value = current_value+increment
            values[i] = new_value
            current_value = new_value
        return values

    @computed_field
    @cached_property
    def initial_endowment(self) -> float:
        return sum(self.values.values())

class EconomicAgent(BaseModel):
    id: str
    endowment: Endowment
    value_schedules: Dict[str, BuyerPreferenceSchedule] = Field(default_factory=dict)
    cost_schedules: Dict[str, SellerPreferenceSchedule] = Field(default_factory=dict)
    max_relative_spread: float = Field(default=0.2)
    pending_orders: Dict[str, List[MarketAction]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_schedules(self):
        overlapping_goods = set(self.value_schedules.keys()) & set(self.cost_schedules.keys())
        if overlapping_goods:
            raise ValueError(f"Agent cannot be both buyer and seller of the same good(s): {overlapping_goods}")
        return self
    
    @computed_field
    @property
    def initial_utility(self) -> float:
        utility = self.endowment.initial_basket.cash
        for good, quantity in self.endowment.initial_basket.goods_dict.items():
            if self.is_buyer(good):
                # Buyers start with cash and no goods
                pass
            elif self.is_seller(good):
                schedule = self.cost_schedules[good]
                initial_quantity = int(quantity)
                initial_cost = sum(schedule.get_value(q) for q in range(1, initial_quantity + 1))
                utility += initial_cost  # Add total cost of initial inventory
        return utility

    @computed_field
    @property
    def current_utility(self) -> float:
        return self.calculate_utility(self.endowment.current_basket)

    @computed_field
    @property
    def current_cash(self) -> float:
        return self.endowment.current_basket.cash
    
    @computed_field
    @property
    def pending_cash(self) -> float:
        return sum(order.price * order.quantity for orders in self.pending_orders.values() for order in orders if isinstance(order, Bid))
    
    @computed_field
    @property
    def available_cash(self) -> float:
        return self.current_cash - self.pending_cash
    
    
    def get_pending_bid_quantity(self, good_name: str) -> int:
        """ returns the quantity of the good that the agent has pending orders for"""
        return sum(order.quantity for order in self.pending_orders.get(good_name, []) if isinstance(order, Bid))
    
    def get_pending_ask_quantity(self, good_name: str) -> int:
        """ returns the quantity of the good that the agent has pending orders for"""
        return sum(order.quantity for order in self.pending_orders.get(good_name, []) if isinstance(order, Ask))
    
    def get_quantity_for_bid(self, good_name: str) -> Optional[int]:
        if not self.is_buyer(good_name):
            return None
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        pending_quantity = self.get_pending_bid_quantity(good_name)
        total_quantity = int(current_quantity + pending_quantity)
        if total_quantity > self.value_schedules[good_name].num_units:
            return None
        print(f"total_quantity: {total_quantity}")
        return total_quantity+1

    
    def get_current_value(self, good_name: str) -> Optional[float]:
        """ returns the current value of the next unit of the good  for buyers only
        it takes in consideration both the current inventory and the pending orders"""
        total_quantity = self.get_quantity_for_bid(good_name)
        if total_quantity is None:
            return None
        current_value = self.value_schedules[good_name].get_value(total_quantity)
        return current_value if current_value  > 0 else None
    
    def get_previous_value(self, good_name: str) -> Optional[float]:
        """ returns the previous value of the good for buyers only
        it takes in consideration both the current inventory and the pending orders"""
        total_quantity = self.get_quantity_for_bid(good_name)
        if total_quantity is None:
            return None
        value = self.value_schedules[good_name].get_value(total_quantity+1)
        return value if value > 0 else None
    
    
    def get_current_cost(self, good_name: str) -> Optional[float]:
        """ returns the current cost of the good for sellers only
        it takes in consideration both the current inventory and the pending orders"""
        if not self.is_seller(good_name):
            return None
        starting_quantity = self.endowment.initial_basket.get_good_quantity(good_name)
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        pending_quantity = self.get_pending_ask_quantity(good_name)
        total_quantity = int(starting_quantity - current_quantity + pending_quantity)+1
        cost = self.cost_schedules[good_name].get_value(total_quantity)

        return cost if cost > 0 else None
    def get_previous_cost(self, good_name: str) -> Optional[float]:
        """ returns the previous cost of the good for sellers only
        it takes in consideration both the current inventory and the pending orders"""
        if not self.is_seller(good_name):
            return None
        starting_quantity = self.endowment.initial_basket.get_good_quantity(good_name)
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        pending_quantity = self.get_pending_ask_quantity(good_name)
        total_quantity = int(starting_quantity - current_quantity + pending_quantity)
        return self.cost_schedules[good_name].get_value(total_quantity)

    def is_buyer(self, good_name: str) -> bool:
        return good_name in self.value_schedules

    def is_seller(self, good_name: str) -> bool:
        return good_name in self.cost_schedules

    def generate_bid(self, good_name: str) -> Optional[Bid]:
        if not self._can_generate_bid(good_name):
            return None
        price = self._calculate_bid_price(good_name)
        if price is not None:
            bid = Bid(price=price, quantity=1)
            # Update pending orders
            self.pending_orders.setdefault(good_name, []).append(bid)
            return bid
        else:
            return None

    def generate_ask(self, good_name: str) -> Optional[Ask]:
        if not self._can_generate_ask(good_name):
            return None
        price = self._calculate_ask_price(good_name)
        if price is not None:
            ask = Ask(price=price, quantity=1)
            # Update pending orders
            self.pending_orders.setdefault(good_name, []).append(ask)
            return ask
        else:
            return None
    
    def would_accept_trade(self, trade: Trade) -> bool:
        if self.is_buyer(trade.good_name) and trade.buyer_id == self.id:
            marginal_value = self.get_previous_value(trade.good_name)
            if marginal_value is None:
                print("trade rejected because marginal_value is None")
                return False
            print(f"Buyer {self.id} would accept trade with marginal_value: {marginal_value}, trade.price: {trade.price}")
            return marginal_value >= trade.price
        elif self.is_seller(trade.good_name) and trade.seller_id == self.id:
            marginal_cost = self.get_previous_cost(trade.good_name)
            if marginal_cost is None:
                print("trade rejected because marginal_cost is None")
                return False
            print(f"Seller {self.id} would accept trade with marginal_cost: {marginal_cost}, trade.price: {trade.price}")
            return trade.price >= marginal_cost
        else:
            self_type = "buyer" if self.is_buyer(trade.good_name) else "seller"
            print(f"trade rejected because it's not for the agent {self.id} vs trade.buyer_id: {trade.buyer_id}, trade.seller_id: {trade.seller_id} with type {self_type}")
            return False

    def process_trade(self, trade: Trade):
        if self.is_buyer(trade.good_name) and trade.buyer_id == self.id:
            # Find the exactly matching bid from pending_orders
            bids = self.pending_orders.get(trade.good_name, [])
            matching_bid = next((bid for bid in bids if bid.quantity == trade.quantity and bid.price == trade.bid_price), None)
            if matching_bid:
                bids.remove(matching_bid)
                if not bids:
                    del self.pending_orders[trade.good_name]
            else:
                raise ValueError(f"Trade {trade.trade_id} processed but matching bid not found for agent {self.id}")
        elif self.is_seller(trade.good_name) and trade.seller_id == self.id:
            # Find the exactly matching ask from pending_orders
            asks = self.pending_orders.get(trade.good_name, [])
            matching_ask = next((ask for ask in asks if ask.quantity == trade.quantity and ask.price == trade.ask_price), None)
            if matching_ask:
                asks.remove(matching_ask)
                if not asks:
                    del self.pending_orders[trade.good_name]
            else:
                raise ValueError(f"Trade {trade.trade_id} processed but matching ask not found for agent {self.id}")
        
        # Only update the endowment after passing the value error checks
        self.endowment.add_trade(trade)
        new_utility = self.calculate_utility(self.endowment.current_basket)
        logger.info(f"Agent {self.id} processed trade. New utility: {new_utility:.2f}")

    def reset_pending_orders(self,good_name:str):
        self.pending_orders[good_name] = []

    def reset_all_pending_orders(self):
        self.pending_orders = {}
        

    def calculate_utility(self, basket: Basket) -> float:
        utility = basket.cash
        
        for good, quantity in basket.goods_dict.items():
            if self.is_buyer(good):
                schedule = self.value_schedules[good]
                value_sum = sum(schedule.get_value(q) for q in range(1, int(quantity) + 1))
                utility += value_sum
            elif self.is_seller(good):
                schedule = self.cost_schedules[good]
                starting_quantity = self.endowment.initial_basket.get_good_quantity(good)
                unsold_units = int(basket.get_good_quantity(good))
                sold_units = starting_quantity - unsold_units
                # Unsold inventory should be valued at its cost, not higher
                unsold_cost = sum(schedule.get_value(q) for q in range(sold_units+1, starting_quantity + 1))

                utility += unsold_cost  # Add the cost of unsold units
        return utility



    def calculate_individual_surplus(self) -> float:
        current_utility = self.calculate_utility(self.endowment.current_basket)
        surplus = current_utility - self.initial_utility
        print(f"current_utility: {current_utility}, initial_utility: {self.initial_utility}, surplus: {surplus}")
        return surplus



    def _can_generate_bid(self, good_name: str) -> bool:
        if not self.is_buyer(good_name):
            return False

        available_cash = self.available_cash 
        if available_cash <= 0:
            return False
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        pending_quantity = sum(order.quantity for order in self.pending_orders.get(good_name, []) if isinstance(order, Bid))
        total_quantity = current_quantity + pending_quantity
        return total_quantity < self.value_schedules[good_name].num_units

    def _can_generate_ask(self, good_name: str) -> bool:
        if not self.is_seller(good_name):
            return False
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        pending_quantity = sum(order.quantity for order in self.pending_orders.get(good_name, []) if isinstance(order, Ask))
        total_quantity = current_quantity - pending_quantity
        return total_quantity > 0

    def _calculate_bid_price(self, good_name: str) -> Optional[float]:

        current_value = self.get_current_value(good_name)
        if current_value is None:
            return None
        max_bid = min(self.endowment.current_basket.cash, current_value*0.99)
        price = random.uniform(max_bid * (1 - self.max_relative_spread), max_bid)
        return price

    def _calculate_ask_price(self, good_name: str) -> Optional[float]:
        current_cost = self.get_current_cost(good_name)
        if current_cost is None:
            return None
        min_ask = current_cost * 1.01
        price = random.uniform(min_ask, min_ask * (1 + self.max_relative_spread))
        return price

    def print_status(self):
        print(f"\nAgent ID: {self.id}")
        print(f"Current Endowment:")
        print(f"  Cash: {self.endowment.current_basket.cash:.2f}")
        print(f"  Goods: {self.endowment.current_basket.goods_dict}")
        current_utility = self.calculate_utility(self.endowment.current_basket)
        print(f"Current Utility: {current_utility:.2f}")
        self.calculate_individual_surplus()

def create_economic_agent(
    agent_id: str,
    goods: List[str],
    buy_goods: List[str],
    sell_goods: List[str],
    base_values: Dict[str, float],
    initial_cash: float,
    initial_goods: Dict[str, int],
    num_units: int = 20,
    noise_factor: float = 0.1,
    max_relative_spread: float = 0.2,
) -> EconomicAgent:
    initial_goods_list = [Good(name=name, quantity=quantity) for name, quantity in initial_goods.items()]
    initial_basket = Basket(
        cash=initial_cash,
        goods=initial_goods_list
    )
    endowment = Endowment(
        initial_basket=initial_basket,
        agent_id=agent_id
    )

    value_schedules = {
        good: BuyerPreferenceSchedule(
            num_units=num_units,
            base_value=base_values[good],
            noise_factor=noise_factor
        ) for good in buy_goods
    }
    cost_schedules = {
        good: SellerPreferenceSchedule(
            num_units=num_units,
            base_value=base_values[good] * 0.7,  # Set seller base value lower than buyer base value
            noise_factor=noise_factor
        ) for good in sell_goods
    }

    return EconomicAgent(
        id=agent_id,
        endowment=endowment,
        value_schedules=value_schedules,
        cost_schedules=cost_schedules,
        max_relative_spread=max_relative_spread
    )



class MarketAgentPromptManager(BaseModel):
    prompts: Dict[str, str] = Field(default_factory=dict)
    prompt_file: str = Field(default="./market_agent/market_agent/market_agent_prompt.yaml")

    def __init__(self, **data: Any):
        super().__init__(**data)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        full_path = os.path.join(project_root, self.prompt_file)
        
        try:
            with open(full_path, 'r') as file:
                self.prompts = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {full_path}")

    def format_prompt(self, prompt_type: str, variables: Dict[str, Any]) -> str:
        if prompt_type not in self.prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        prompt_template = self.prompts[prompt_type]
        return prompt_template.format(**variables)

    def get_perception_prompt(self, variables: Dict[str, Any]) -> str:
        return self.format_prompt('perception', variables)

    def get_action_prompt(self, variables: Dict[str, Any]) -> str:
        return self.format_prompt('action', variables)

    def get_reflection_prompt(self, variables: Dict[str, Any]) -> str:
        return self.format_prompt('reflection', variables)

class AgentPromptVariables(BaseModel):
    environment_name: str
    environment_info: Any
    recent_memories: List[Dict[str, Any]] = Field(default_factory=list)
    perception: Optional[Any] = None
    observation: Optional[Any] = None
    action_space: Dict[str, Any] = {}
    last_action: Optional[Any] = None
    reward: Optional[float] = None
    previous_strategy: Optional[str] = None

class Persona(BaseModel):
    name: str
    role: str
    persona: str
    objectives: List[str]
    trader_type: List[str] 

class SystemPromptSchema(BaseModel):
    """Schema for system prompts."""
    Role: str
    Persona: Optional[str] = None
    Objectives: Optional[str] = None

class PromptManager:
    """
    Manages the creation and formatting of prompts for AI agents.

    This class handles loading prompt templates, formatting prompts with variables,
    and generating system and task prompts for AI agent interactions.
    """

    def __init__(self, role: str, task: Union[str, List[str]], persona: Optional[str] = None, objectives: Optional[str] = None, 
                 resources: Optional[Any] = None, output_schema: Optional[Union[str, Dict[str, Any]]] = None, 
                 char_limit: Optional[int] = None):
        """
        Initialize the PromptManager.

        Args:
            role (str): The role of the AI agent.
            task (Union[str, List[str]]): The task or list of tasks for the agent.
            persona (Optional[str]): The persona characteristics for the agent.
            objectives (Optional[str]): The objectives of the agent.
            resources (Optional[Any]): Additional resources for prompt generation.
            output_schema (Optional[Union[str, Dict[str, Any]]]): The schema for the expected output or output format.
            char_limit (Optional[int]): Character limit for the prompts.
        """
        self.role = role
        self.persona = persona
        self.objectives = objectives
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.char_limit = char_limit
        self.prompt_vars = self._create_prompt_vars_dict(task, resources, output_schema)
        self.prompt_path = os.path.join(self.script_dir, f"{self.role}_prompt.yaml")
        self.default_prompt_path = os.path.join(self.script_dir, 'market_agent_prompt.yaml')
        self.system_prompt_schema, self.task_prompt_schema = self._read_yaml_file(self.prompt_path)

    def format_yaml_prompt(self) -> str:
        """
        Format the YAML prompt with variables.

        Returns:
            str: Formatted YAML prompt.
        """
        formatted_prompt = ""
        for field, value in self.system_prompt_schema.dict().items():
            formatted_value = value.format(**self.prompt_vars.dict()) if value else ""
            formatted_prompt += f"# {field}:\n{formatted_value}\n"
        for field, value in self.task_prompt_schema.dict().items():
            formatted_value = value.format(**self.prompt_vars.dict()) if value else ""
            formatted_prompt += f"# {field}:\n{formatted_value}\n"
        return formatted_prompt

    def _read_yaml_file(self, file_path: Optional[str] = None) -> Tuple[SystemPromptSchema, TaskPromptSchema]:
        """
        Read and parse a YAML file.

        Args:
            file_path (Optional[str]): Path to the YAML file.

        Returns:
            Tuple[SystemPromptSchema, TaskPromptSchema]: Parsed YAML content as SystemPromptSchema and TaskPromptSchema.

        Raises:
            FileNotFoundError: If neither the specified file nor the default file is found.
            ValueError: If there's an error parsing the YAML file.
        """
        try:
            with open(file_path, 'r') as file:
                yaml_content = yaml.safe_load(file)
        except FileNotFoundError:
            try:
                with open(self.default_prompt_path, 'r') as file:
                    yaml_content = yaml.safe_load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f"Neither the role-specific prompt file at {file_path} "
                                        f"nor the default prompt file at {self.default_prompt_path} were found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

        system_prompt_data = {k: v for k, v in yaml_content.items() if k in SystemPromptSchema.__fields__}
        task_prompt_data = {k: v for k, v in yaml_content.items() if k in TaskPromptSchema.__fields__}

        return SystemPromptSchema(**system_prompt_data), TaskPromptSchema(**task_prompt_data)

    def _create_prompt_vars_dict(self, task: Union[str, List[str]], resources: Optional[Any],
                          output_schema: Optional[Union[str, Dict[str, Any]]]) -> PromptTemplateVariables:
        """
        Create a dictionary of variables for prompt formatting.

        Args:
            task (Union[str, List[str]]): The task or list of tasks.
            resources (Optional[Any]): Additional resources.
            output_schema (Optional[Union[str, Dict[str, Any]]]): The output schema or format.

        Returns:
            PromptTemplateVariables: Pydantic model of variables for prompt formatting.

        Raises:
            NotImplementedError: If document retrieval is not implemented.
        """
        documents = ""
        if resources is not None:
            # TODO: @interstellarninja implement document/memory retrieval methods
            raise NotImplementedError("Document retrieval is not implemented yet.")
        
        pydantic_schema = output_schema if isinstance(output_schema, dict) else None
        output_format = output_schema if isinstance(output_schema, str) else "json_object"
        
        input_vars = PromptTemplateVariables(
            role=self.role,
            persona=self.persona,
            objectives=self.objectives,
            task=task,
            datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            doc_list=documents,
            pydantic_schema=json.dumps(pydantic_schema) if pydantic_schema else None,
            output_format=output_format
        )

        return input_vars

    def generate_system_prompt(self) -> str:
        """
        Generate the system prompt.

        Returns:
            str: Formatted system prompt.
        """
        system_content = f"Role: {self.system_prompt_schema.Role.format(**self.prompt_vars.dict())}\n"
        
        if self.persona and self.system_prompt_schema.Persona:
            system_content += f"Persona: {self.system_prompt_schema.Persona.format(**self.prompt_vars.dict())}\n"
        
        if self.objectives and self.system_prompt_schema.Objectives:
            system_content += f"Objectives: {self.system_prompt_schema.Objectives.format(**self.prompt_vars.dict())}\n"
        
        return system_content

    def generate_task_prompt(self) -> str:
        """
        Generate the task prompt.

        Returns:
            str: Formatted task prompt.
        """
        user_content = f"Tasks: {self.task_prompt_schema.Tasks.format(**self.prompt_vars.dict())}\n"
        
        if self.prompt_vars.pydantic_schema and self.task_prompt_schema.Output_schema:
            user_content += f"Output_schema: {self.task_prompt_schema.Output_schema.format(**self.prompt_vars.dict())}\n"
        else:
            user_content += f"Output_format: {self.prompt_vars.output_format}\n"
        
        if self.task_prompt_schema.Assistant:
            user_content += f"Assistant: {self.task_prompt_schema.Assistant.format(**self.prompt_vars.dict())}"
        
        return user_content

    def generate_prompt_messages(self, system_prefix: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate prompt messages for AI interaction.

        Args:
            system_prefix (Optional[str]): Prefix for the system prompt.

        Returns:
            Dict[str, List[Dict[str, str]]]: Dictionary containing system and user messages.
        """
        system_prompt = self.generate_system_prompt()
        if system_prefix:
            system_prompt = system_prefix + system_prompt
        user_prompt = self.generate_task_prompt()
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

class Agent(BaseModel):
    """Base class for all agents in the multi-agent system.

    Attributes:
        id (str): Unique identifier for the agent.
        role (str): Role of the agent in the system.
        persona(str): Personal characteristics of the agent.
        system (Optional[str]): System instructions for the agent.
        task (Optional[str]): Current task assigned to the agent.
        tools (Optional[Dict[str, Any]]): Tools available to the agent.
        output_format (Optional[Union[Dict[str, Any], str]]): Expected output format.
        llm_config (LLMConfig): Configuration for the language model.
        max_retries (int): Maximum number of retry attempts for AI inference.
        metadata (Optional[Dict[str, Any]]): Additional metadata for the agent.
        interactions (List[Dict[str, Any]]): History of agent interactions.

    Methods:
        execute(task: Optional[str] = None, output_format: Optional[Union[Dict[str, Any], str]] = None, return_prompt: bool = False) -> Union[str, Dict[str, Any], LLMPromptContext]:
            Execute a task and return the result or the prompt context.
        _load_output_schema(output_format: Optional[Union[Dict[str, Any], str]]) -> Optional[Dict[str, Any]]:
            Load the output schema based on the output_format.
        _prepare_prompt_context(task: Optional[str], output_format: Optional[Dict[str, Any]]) -> LLMPromptContext:
            Prepare LLMPromptContext for AI inference.
        _run_ai_inference(prompt_context: LLMPromptContext) -> Union[str, Dict[str, Any]]:
            Run AI inference with retry logic.
        _log_interaction(prompt: LLMPromptContext, response: Union[str, Dict[str, Any]]) -> None:
            Log an interaction between the agent and the AI.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str
    persona: Optional[str] = None
    system: Optional[str] = None
    task: Optional[str] = None
    tools: Optional[Dict[str, Any]] = None
    output_format: Optional[Union[Dict[str, Any], str]] = None
    llm_config: LLMConfig = Field(default_factory=LLMConfig)
    max_retries: int = 2
    metadata: Optional[Dict[str, Any]] = None
    interactions: List[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        extra = "allow"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.ai_utilities = ParallelAIUtilities()

    async def execute(self, task: Optional[str] = None, output_format: Optional[Union[Dict[str, Any], str, Type[BaseModel]]] = None, return_prompt: bool = False) -> Union[str, Dict[str, Any], LLMPromptContext]:
        """Execute a task and return the result or the prompt context."""
        execution_task = task if task is not None else self.task
        if execution_task is None:
            raise ValueError("No task provided. Agent needs a task to execute.")
        
        execution_output_format = output_format if output_format is not None else self.output_format
        
        # Update llm_config based on output_format
        if execution_output_format == "plain_text":
            self.llm_config.response_format = "text"
        else:
            self.llm_config.response_format = "structured_output"
            execution_output_format = self._load_output_schema(execution_output_format)

        prompt_context = self._prepare_prompt_context(
            execution_task,
            execution_output_format if isinstance(execution_output_format, dict) else None
        )
        logger.debug(f"Prepared LLMPromptContext:\n{json.dumps(prompt_context.model_dump(), indent=2)}")
        if return_prompt:
            return prompt_context

        result = await self._run_ai_inference(prompt_context)
        self._log_interaction(prompt_context, result)
        return result

    def _load_output_schema(self, output_format: Optional[Union[Dict[str, Any], str, Type[BaseModel]]] = None) -> Optional[Dict[str, Any]]:
        """Load the output schema based on the output_format."""
        if output_format is None:
            output_format = self.output_format

        if isinstance(output_format, str):
            try:
                schema_class = globals().get(output_format)
                if schema_class and issubclass(schema_class, BaseModel):
                    return schema_class.model_json_schema()
                else:
                    raise ValueError(f"Invalid schema: {output_format}")
            except (AttributeError, ValueError) as e:
                logger.warning(f"Could not load schema: {output_format}. Error: {str(e)}")
                return None
        elif isinstance(output_format, dict):
            return output_format
        elif isinstance(output_format, type) and issubclass(output_format, BaseModel):
            return output_format.model_json_schema()
        else:
            return None

    def _prepare_prompt_context(self, task: Optional[str], output_format: Optional[Dict[str, Any]] = None) -> LLMPromptContext:
        """Prepare LLMPromptContext for AI inference."""
        prompt_manager = PromptManager(
            role=self.role,
            persona=self.persona,
            task=task if task is not None else [],
            resources=None,
            output_schema=output_format,
            char_limit=1000
        )

        prompt_messages = prompt_manager.generate_prompt_messages()
        system_message = prompt_messages["messages"][0]["content"]
        if self.system:
            system_message += f"\n{self.system}"
        user_message = prompt_messages["messages"][1]["content"]
       
        structured_output = None
        if output_format and isinstance(output_format, dict):
            structured_output = StructuredTool(json_schema=output_format, strict_schema=False)

        return LLMPromptContext(
            id=self.id,
            system_string=system_message,
            new_message=user_message,
            llm_config=self.llm_config,
            structured_output=structured_output
        )
    
    @retry(
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(2),
        reraise=True
    )
    async def _run_ai_inference(self, prompt_context: LLMPromptContext) -> Any:
        try:
            llm_output = await self.ai_utilities.run_parallel_ai_completion([prompt_context])
            
            if not llm_output:
                raise ValueError("No output received from AI inference")
            
            llm_output = llm_output[0]
            
            if prompt_context.llm_config.response_format == "text":
                return llm_output.str_content or str(llm_output.raw_result)
            elif prompt_context.llm_config.response_format in ["json_beg", "json_object", "structured_output"]:
                if llm_output.json_object:
                    return llm_output.json_object.object
                elif llm_output.str_content:
                    try:
                        return json.loads(llm_output.str_content)
                    except json.JSONDecodeError:
                        return extract_json_from_response(llm_output.str_content)
            elif prompt_context.llm_config.response_format == "tool":
                # Handle tool response format if needed
                pass
            
            # If no specific handling or parsing failed, return the raw output
            logger.warning(f"No parsing logic for response format '{prompt_context.llm_config.response_format}'. Returning raw output.")
            return llm_output.raw_result
        
        except Exception as e:
            logger.error(f"Error during AI inference: {e}")
            raise

    def _log_interaction(self, prompt: LLMPromptContext, response: Union[str, Dict[str, Any]]) -> None:
        """Log an interaction between the agent and the AI."""
        interaction = {
            "id": self.id,
            "name": self.role,
            "system": prompt.system_message,
            "task": prompt.new_message,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        self.interactions.append(interaction)
        logger.debug(f"Agent Interaction logged:\n{json.dumps(interaction, indent=2)}")


class Performative(Enum):
    """Enumeration of ACL message performatives."""
    ACCEPT_PROPOSAL = "accept-proposal"
    CALL_FOR_PROPOSAL = "cfp"
    PROPOSE = "propose"
    REJECT_PROPOSAL = "reject-proposal"
    INFORM = "inform"
    REQUEST = "request"
    QUERY_IF = "query-if"
    NOT_UNDERSTOOD = "not-understood"

class AgentID(BaseModel):
    """Represents the identity of an agent."""
    name: str
    address: Optional[str] = None

class ACLMessage(Protocol, BaseModel, Generic[T]):
    """Represents an ACL message with various attributes and creation methods."""
    performative: Optional[Performative] = None
    sender: Optional[AgentID] = None
    receivers: Optional[List[AgentID]] = None
    content: Optional[T] = None
    reply_with: Optional[str] = None
    in_reply_to: Optional[str] = None
    conversation_id: Optional[str] = None
    protocol: str = "double-auction"
    language: str = "JSON"
    ontology: str = "market-ontology"
    reply_by: Optional[datetime] = None

    class Config:
        use_enum_values = True

    @classmethod
    def create_observation(cls, sender: str, agent_id: str, content: Any, step: int):
        """Create an observation message."""
        return cls(
            performative=Performative.INFORM,  # Use the enum value directly
            sender=AgentID(name=sender),
            receivers=[AgentID(name=agent_id)],
            content=content,
            protocol="double-auction",
            ontology="market-ontology",
            conversation_id=f"observation-{step}-{agent_id}"
        )

    def parse_action(self) -> Dict[str, Any]:
        """Parse the ACL message content into a market action."""
        if self.performative not in [Performative.PROPOSE, Performative.REQUEST]:
            return {"type": "hold", "price": 0, "quantity": 0}
        
        content = self.content
        if not isinstance(content, dict):
            return {"type": "hold", "price": 0, "quantity": 0}
        
        action_type = content.get("type", "hold")
        if action_type not in ["bid", "ask", "hold"]:
            return {"type": "hold", "price": 0, "quantity": 0}
        
        return {
            "type": action_type,
            "price": float(content.get("price", 0)),
            "quantity": int(content.get("quantity", 0))
        }

    def generate_message(self, *args, **kwargs) -> 'ACLMessage':
        """
        Generate a new ACLMessage based on the provided arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            ACLMessage: A new ACLMessage instance.
        """
        return self.create_message(*args, **kwargs)

    @classmethod
    def create_bid(cls, sender: AgentID, receiver: AgentID, bid_price: float, bid_quantity: int) -> 'ACLMessage':
        """
        Create a bid message.

        Args:
            sender (AgentID): The agent sending the bid.
            receiver (AgentID): The agent receiving the bid.
            bid_price (float): The price of the bid.
            bid_quantity (int): The quantity of the bid.

        Returns:
            ACLMessage: A new ACLMessage instance representing a bid.
        """
        content = {
            "type": "bid",
            "price": bid_price,
            "quantity": bid_quantity
        }
        return cls(
            performative=Performative.PROPOSE,
            sender=sender,
            receivers=[receiver],
            content=content,
            conversation_id=f"bid-{datetime.now().isoformat()}"
        )

    @classmethod
    def create_ask(cls, sender: AgentID, receiver: AgentID, ask_price: float, ask_quantity: int) -> 'ACLMessage':
        """
        Create an ask message.

        Args:
            sender (AgentID): The agent sending the ask.
            receiver (AgentID): The agent receiving the ask.
            ask_price (float): The price of the ask.
            ask_quantity (int): The quantity of the ask.

        Returns:
            ACLMessage: A new ACLMessage instance representing an ask.
        """
        content = {
            "type": "ask",
            "price": ask_price,
            "quantity": ask_quantity
        }
        return cls(
            performative=Performative.PROPOSE,
            sender=sender,
            receivers=[receiver],
            content=content,
            conversation_id=f"ask-{datetime.now().isoformat()}"
        )

    @classmethod
    def create_accept(cls, sender: AgentID, receiver: AgentID, original_message_id: str) -> 'ACLMessage':
        """
        Create an accept message in response to a proposal.

        Args:
            sender (AgentID): The agent sending the accept message.
            receiver (AgentID): The agent receiving the accept message.
            original_message_id (str): The ID of the original proposal being accepted.

        Returns:
            ACLMessage: A new ACLMessage instance representing an acceptance.
        """
        return cls(
            performative=Performative.ACCEPT_PROPOSAL,
            sender=sender,
            receivers=[receiver],
            content={"accepted": True},
            in_reply_to=original_message_id
        )

    @classmethod
    def create_reject(cls, sender: AgentID, receiver: AgentID, original_message_id: str, reason: str) -> 'ACLMessage':
        """
        Create a reject message in response to a proposal.

        Args:
            sender (AgentID): The agent sending the reject message.
            receiver (AgentID): The agent receiving the reject message.
            original_message_id (str): The ID of the original proposal being rejected.
            reason (str): The reason for rejection.

        Returns:
            ACLMessage: A new ACLMessage instance representing a rejection.
        """
        return cls(
            performative=Performative.REJECT_PROPOSAL,
            sender=sender,
            receivers=[receiver],
            content={"accepted": False, "reason": reason},
            in_reply_to=original_message_id
        )

    @classmethod
    def create_inform(cls, sender: AgentID, receiver: AgentID, info_type: str, info_content: Any) -> 'ACLMessage':
        """
        Create an inform message.

        Args:
            sender (AgentID): The agent sending the inform message.
            receiver (AgentID): The agent receiving the inform message.
            info_type (str): The type of information being sent.
            info_content (Any): The content of the information.

        Returns:
            ACLMessage: A new ACLMessage instance representing an inform message.
        """
        return cls(
            performative=Performative.INFORM,
            sender=sender,
            receivers=[receiver],
            content={"type": info_type, "content": info_content}
        )

    @classmethod
    def create_message(cls, performative: Performative, sender: str, receiver: str, content: Any, **kwargs) -> 'ACLMessage':
        """
        Create a generic ACL message.

        Args:
            performative (Performative): The performative of the message.
            sender (str): The name of the sender agent.
            receiver (str): The name of the receiver agent.
            content (Any): The content of the message.
            **kwargs: Additional keyword arguments for the message.

        Returns:
            ACLMessage: A new ACLMessage instance.
        """
        return cls(
            performative=performative,
            sender=AgentID(name=sender),
            receivers=[AgentID(name=receiver)],
            content=content,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ACLMessage to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the ACLMessage.
        """
        return self.dict(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ACLMessage':
        """
        Create an ACLMessage instance from a dictionary.

        Args:
            data (Dict[str, Any]): A dictionary containing ACLMessage data.

        Returns:
            ACLMessage: A new ACLMessage instance created from the dictionary data.
        """
        return cls(**data)
    
    def parse_to_market_action(self) -> Dict[str, Any]:
        """Parse the ACL message content into a market action."""
        if self.performative not in [Performative.PROPOSE, Performative.REQUEST]:
            return {"type": "hold", "price": 0, "quantity": 0}
        
        content = self.content
        if not isinstance(content, dict):
            return {"type": "hold", "price": 0, "quantity": 0}
        
        action_type = content.get("type", "hold")
        if action_type not in ["bid", "ask", "hold"]:
            return {"type": "hold", "price": 0, "quantity": 0}
        
        return {
            "type": action_type,
            "price": float(content.get("price", 0)),
            "quantity": int(content.get("quantity", 0))
        }

class MarketAgent(Agent, EconomicAgent):
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    last_action: Optional[Dict[str, Any]] = None
    last_observation: Optional[LocalObservation] = Field(default_factory=dict)
    environments: Dict[str, MultiAgentEnvironment] = Field(default_factory=dict)
    protocol: Type[Protocol] = Field(..., description="Communication protocol class")
    address: str = Field(default="", description="Agent's address")
    prompt_manager: MarketAgentPromptManager = Field(default_factory=lambda: MarketAgentPromptManager())

    @classmethod
    def create(
        cls,
        agent_id: int,
        is_buyer: bool,
        num_units: int,
        base_value: float,
        use_llm: bool,
        initial_cash: float,
        initial_goods: int,
        good_name: str,
        noise_factor: float = 0.1,
        max_relative_spread: float = 0.2,
        llm_config: Optional[LLMConfig] = None,
        environments: Dict[str, MultiAgentEnvironment] = {},
        protocol: Optional[Type[Protocol]] = None,
        persona: Optional[Persona] = None
    ) -> 'MarketAgent':
    
        econ_agent = create_economic_agent(
            agent_id=str(agent_id),
            goods=[good_name],
            buy_goods=[good_name] if is_buyer else [],
            sell_goods=[] if is_buyer else [good_name],
            base_values={good_name: base_value},
            initial_cash=initial_cash,
            initial_goods={good_name: initial_goods},
            num_units=num_units,
            noise_factor=noise_factor,
            max_relative_spread=max_relative_spread
        )
    
        role = "buyer" if is_buyer else "seller"
        llm_agent = Agent(
            id=str(agent_id),
            role=role,
            persona=persona.persona if persona else None,
            objectives=persona.objectives if persona else None,
            llm_config=llm_config or LLMConfig()
        )

        return cls(
            id=str(agent_id),
            preference_schedule=econ_agent.value_schedules[good_name] if is_buyer else econ_agent.cost_schedules[good_name],
            endowment=econ_agent.endowment,
            utility_function=econ_agent.calculate_utility,
            max_relative_spread=econ_agent.max_relative_spread,
            use_llm=use_llm, 
            address=f"agent_{agent_id}_address",
            environments=environments or {},
            protocol=protocol,  # This should be the ACLMessage class, not an instance
            role=llm_agent.role,
            persona=llm_agent.persona,
            objectives=llm_agent.objectives,
            llm_config=llm_agent.llm_config,
        )

    async def perceive(self, environment_name: str, return_prompt: bool = False) -> Union[str, LLMPromptContext]:
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")

        environment_info = self.environments[environment_name].get_global_state()
        recent_memories = self.memory[-5:] if self.memory else []
        
        variables = AgentPromptVariables(
            environment_name=environment_name,
            environment_info=environment_info,
            recent_memories=recent_memories if recent_memories else []
        )
        
        prompt = self.prompt_manager.get_perception_prompt(variables.model_dump())
        
        return await self.execute(prompt, output_format=PerceptionSchema.model_json_schema(), return_prompt=return_prompt)

    async def generate_action(self, environment_name: str, perception: Optional[str] = None, return_prompt: bool = False) -> Union[Dict[str, Any], LLMPromptContext]:
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")

        environment = self.environments[environment_name]
        if perception is None and not return_prompt:
            perception = await self.perceive(environment_name)
        environment_info = environment.get_global_state()
        action_space = environment.action_space
        serialized_action_space = {
            "allowed_actions": [action_type.__name__ for action_type in action_space.allowed_actions]
        }
        
        variables = AgentPromptVariables(
            environment_name=environment_name,
            environment_info=environment_info,
            perception=perception,
            action_space=serialized_action_space,
            last_action=self.last_action,
            observation=self.last_observation
        )
        
        prompt = self.prompt_manager.get_action_prompt(variables.model_dump())
     
        action_schema = action_space.get_action_schema()
        
        response = await self.execute(prompt, output_format=action_schema, return_prompt=return_prompt)
        
        if not return_prompt:
            action = {
                "sender": self.id,
                "content": response,
            }
            self.last_action = response
            return action
        else:
            return response

    async def reflect(self, environment_name: str, return_prompt: bool = False) -> Union[str, LLMPromptContext]:
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")
        
        environment = self.environments[environment_name]
        last_step = environment.history.steps[-1][1] if environment.history.steps else None
        
        if last_step:
            local_step = last_step.get_local_step(self.id)
            observation = local_step.observation.observation
            reward = local_step.info.get('reward', 0)
        else:
            observation = {}
            reward = 0

        environment_info = environment.get_global_state()
        previous_strategy = self.memory[-1].get('strategy_update', 'No previous strategy') if self.memory else 'No previous strategy'
        
        variables = AgentPromptVariables(
            environment_name=environment_name,
            environment_info=environment_info,
            observation=observation,
            last_action=self.last_action,
            reward=reward,
            previous_strategy=previous_strategy
        )
        
        prompt = self.prompt_manager.get_reflection_prompt(variables.model_dump())
        
        response = await self.execute(prompt, output_format=ReflectionSchema.model_json_schema(), return_prompt=return_prompt)
        
        if not return_prompt:
            if isinstance(response, dict):
                reflection = response.get("reflection", "")
                strategy_update = response.get("strategy_update")
            else:
                reflection = response
                strategy_update = None

            self.memory.append({
                "type": "reflection",
                "content": reflection,
                "strategy_update": strategy_update,
                "observation": observation,
                "reward": reward,
                "timestamp": datetime.now().isoformat()
            })
            return reflection
        else:
            return response

class GroupChatMessage(BaseModel):
    content: str
    message_type: Literal["propose_topic", "group_message"]
    agent_id: str

class GroupChatAction(LocalAction):
    action: GroupChatMessage

    @classmethod
    def sample(cls, agent_id: str) -> 'GroupChatAction':
        return cls(
            agent_id=agent_id, 
            action=GroupChatMessage(
                content="Sample message", 
                message_type="group_message",
                agent_id=agent_id
            )
        )

    @classmethod
    def action_schema(cls) -> Type[BaseModel]:
        return cls.model_json_schema()

class GroupChatGlobalAction(GlobalAction):
    actions: Dict[str, Dict[str, Any]]

class GroupChatObservation(BaseModel):
    messages: List[GroupChatMessage]
    current_topic: str
    current_speaker: str

class GroupChatLocalObservation(LocalObservation):
    observation: GroupChatObservation

class GroupChatGlobalObservation(GlobalObservation):
    observations: Dict[str, GroupChatLocalObservation]
    all_messages: List[GroupChatMessage]
    current_topic: str
    speaker_order: List[str]

class GroupChat(Mechanism):
    max_rounds: int = Field(..., description="Maximum number of chat rounds")
    current_round: int = Field(default=0, description="Current round number")
    messages: List[GroupChatMessage] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    current_topic: str = Field(..., description="Current discussion topic")
    speaker_order: List[str] = Field(default_factory=list)
    current_speaker_index: int = Field(default=0)
    actions: GroupChatGlobalAction = Field(default=None)

    sequential: bool = Field(default=False, description="Whether the mechanism is sequential")

    def step(self, action: Union[GroupChatAction, GroupChatGlobalAction, Dict[str, Any]]) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        logger.debug(f"Received action of type: {type(action).__name__}")
        logger.debug(f"Action content: {action}")

        if self.sequential:
            # Sequential mode: expect a LocalAction
            if isinstance(action, dict):
                try:
                    action = GroupChatAction.parse_obj(action)
                    logger.debug("Parsed action into GroupChatAction.")
                except Exception as e:
                    logger.error(f"Failed to parse action into GroupChatAction: {e}")
                    raise
            if not isinstance(action, GroupChatAction):
                logger.error(f"Expected GroupChatAction, got {type(action).__name__}")
                raise TypeError(f"Expected GroupChatAction, got {type(action).__name__}")

            # Check if it's the current agent's turn
            if action.agent_id != self.speaker_order[self.current_speaker_index]:
                raise ValueError(f"It's not agent {action.agent_id}'s turn to speak.")

            self.current_round += 1
            logger.debug(f"Processing round {self.current_round} with action: {action}")

            # Process the action
            self.messages.append(action.action)

            # Update topic if necessary
            if action.action.message_type == "propose_topic":
                self._update_topic(action.action.content)

            # Create observation for the agent
            observation = self._create_observation([action.action], action.agent_id)
            done = self.current_round >= self.max_rounds

            # Update the current speaker
            self._select_next_speaker()
            logger.debug(f"Next speaker selected: {self.speaker_order[self.current_speaker_index]}")

            local_step = LocalEnvironmentStep(
                observation=observation,
                reward=0,  # Implement reward function if needed
                done=done,
                info={
                    "current_round": self.current_round,
                    "current_topic": self.current_topic,
                    "all_messages": [message.dict() for message in self.messages],
                    "speaker_order": self.speaker_order
                }
            )

            return local_step

        else:
            # Non-sequential mode: expect a GlobalAction
            if isinstance(action, dict):
                try:
                    action = GroupChatGlobalAction.parse_obj(action)
                    logger.debug("Parsed actions into GroupChatGlobalAction.")
                except Exception as e:
                    logger.error(f"Failed to parse actions into GroupChatGlobalAction: {e}")
                    raise

            # Ensure action is GroupChatGlobalAction
            if not isinstance(action, GroupChatGlobalAction):
                logger.error(f"Expected GroupChatGlobalAction, got {type(action).__name__}")
                raise TypeError(f"Expected GroupChatGlobalAction, got {type(action).__name__}")

            self.current_round += 1
            logger.debug(f"Processing round {self.current_round} with actions: {action}")

            new_messages = self._process_actions(action)
            self.messages.extend(new_messages)

            observations = self._create_observations(new_messages)
            done = self.current_round >= self.max_rounds

            # Optionally, update topic if a propose_topic message is found
            for message in new_messages:
                if message.message_type == "propose_topic":
                    self._update_topic(message.content)

            # Create global_observation
            global_observation = GroupChatGlobalObservation(
                observations=observations,
                all_messages=self.messages,
                current_topic=self.current_topic,
                speaker_order=self.speaker_order
            )

            # Return an EnvironmentStep with your custom global_observation
            env_step = EnvironmentStep(
                global_observation=global_observation,
                done=done,
                info={
                    "current_round": self.current_round,
                    "current_topic": self.current_topic,
                    "all_messages": [message.dict() for message in self.messages],
                    "speaker_order": self.speaker_order
                }
            )

            return env_step

    def _process_actions(self, global_action: GroupChatGlobalAction) -> List[GroupChatMessage]:
        new_messages = []
        for agent_id, action_dict in global_action.actions.items():
            try:
                action = GroupChatAction.parse_obj(action_dict)
                new_messages.append(action.action)
            except Exception as e:
                logger.error(f"Failed to parse action for agent {agent_id}: {e}")
                continue  # Skip invalid actions
        return new_messages

    def _update_topic(self, new_topic: str):
        self.topics.append(new_topic)
        self.current_topic = new_topic
        logger.info(f"Updated topic to: {new_topic}")

    def _create_observation(self, new_messages: List[GroupChatMessage], agent_id: str) -> GroupChatLocalObservation:
        observation = GroupChatObservation(
            messages=new_messages,
            current_topic=self.current_topic,
            current_speaker=self.speaker_order[self.current_speaker_index]
        )
        return GroupChatLocalObservation(
            agent_id=agent_id,
            observation=observation
        )

    def _create_observations(self, new_messages: List[GroupChatMessage]) -> Dict[str, GroupChatLocalObservation]:
        observations = {}
        for agent_id in self.speaker_order:
            observation = GroupChatObservation(
                messages=new_messages,
                current_topic=self.current_topic,
                current_speaker=self.speaker_order[self.current_speaker_index]
            )
            observations[agent_id] = GroupChatLocalObservation(
                agent_id=agent_id,
                observation=observation
            )
        return observations

    def get_global_state(self) -> Dict[str, Any]:
        return {
            "current_round": self.current_round,
            "messages": [message.dict() for message in self.messages],
            "current_topic": self.current_topic,
            "speaker_order": self.speaker_order,
            "current_speaker_index": self.current_speaker_index
        }

    def reset(self) -> None:
        self.current_round = 0
        self.messages = []
        self.current_speaker_index = 0
        logger.info("GroupChat mechanism has been reset.")

    def _select_next_speaker(self) -> str:
        self.current_speaker_index = (self.current_speaker_index + 1) % len(self.speaker_order)
        return self.speaker_order[self.current_speaker_index]


# env = MultiAgentEnvironment(name="group_chat_config.name", address="group_chat_config.address", max_steps="group_chat_config.max_rounds", action_space="GroupChatActionSpace()", observation_space="GroupChatObservationSpace()", mechanism="group_chat", )

def load_config(config_path: Path = Path("./market_agent/orchestrator_config.yaml")) -> OrchestratorConfig:
    with open(config_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return OrchestratorConfig(**yaml_data)

class GroupChatActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = [GroupChatAction]

class GroupChatObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = [GroupChatLocalObservation]

def create_agent(agent_id: int, persona: Persona, llm_config: LLMConfig, config: OrchestratorConfig, environments: MultiAgentEnvironment):
    agent = MarketAgent.create(
        agent_id=agent_id,
        is_buyer=persona.role.lower() == "buyer",
        num_units=config.agent_config.num_units,
        base_value=config.agent_config.base_value,
        use_llm=config.agent_config.use_llm,
        initial_cash=config.agent_config.buyer_initial_cash if persona.role.lower() == "buyer" else config.agent_config.seller_initial_cash,
        initial_goods=config.agent_config.buyer_initial_goods if persona.role.lower() == "buyer" else config.agent_config.seller_initial_goods,
        good_name=config.agent_config.good_name,
        noise_factor=config.agent_config.noise_factor,
        max_relative_spread=config.agent_config.max_relative_spread,
        llm_config=llm_config,
        protocol=ACLMessage,
        environments=environments,
        persona=persona
    )
    return agent

def run(inputs: InputSchema, *args, **kwargs):
    market_agent_0 = create_agent(inputs.agent_id, inputs.persona, inputs.llm_config, inputs.config, inputs.environments)

    tool_input_class = globals().get(inputs.tool_input_type)
    tool_input = tool_input_class(inputs.tool_input_value)
    method = getattr(market_agent_0, inputs.tool_name, None)

    return asyncio.run(method(tool_input))

if __name__ == "__main__":
    from naptha_sdk.utils import load_yaml
    from market_agent.schemas import InputSchema
    from market_agent.persona import load_or_generate_personas

    cfg_path = "market_agent/component.yaml"
    cfg = load_yaml(cfg_path)

    personas = load_or_generate_personas()

    llm_config=LLMConfig(
                client='openai',
                model='gpt-4o-mini',
                temperature=0.7,
                max_tokens=512,
                use_cache=True
            )
    orchestrator_config = load_config()

    group_chat = GroupChat(
        max_rounds=orchestrator_config.max_rounds,
        current_topic="Placeholder topic",
        speaker_order=["0"]
    )
    environment = MultiAgentEnvironment(
        name="group_chat",
        address="group_chat_address",
        max_steps=orchestrator_config.max_rounds,
        action_space=GroupChatActionSpace(),
        observation_space=GroupChatObservationSpace(),
        mechanism=group_chat
    )

    # You will likely need to change the inputs dict
    inputs = {
        "tool_name": "perceive", 
        "tool_input_type": "str", 
        "tool_input_value": "group_chat",
        "agent_id": 0, 
        "persona": personas[0],
        "llm_config": llm_config,
        "config": orchestrator_config,
        "environments": {"group_chat": environment},
    }
    inputs = InputSchema(**inputs)

    response = run(inputs)
    print(response)
