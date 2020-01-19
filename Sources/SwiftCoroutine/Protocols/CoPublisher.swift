//
//  CoPublisher.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 30.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

public protocol CoPublisher {
    
    associatedtype Output
    typealias OutputResult = Result<Output, Error>
    typealias OutputHandler = (OutputResult) -> Void
    typealias HandlerIdentifier = AnyHashable
    typealias Dispatcher = Coroutine.Dispatcher
    
    func subscribe(with identifier: HandlerIdentifier, handler: @escaping OutputHandler)
    func unsubscribe(_ identifier: HandlerIdentifier) -> OutputHandler?
    func await() throws -> Output
    
}

extension CoPublisher {
    
    @inlinable
    public func subscribe(with identifier: HandlerIdentifier, handler: @escaping () -> Void) {
        subscribe(with: identifier) { _ in handler() }
    }
    
    @inlinable @discardableResult
    public func addHandler(_ handler: @escaping OutputHandler) -> HandlerIdentifier {
        let identifier = withUnsafeBytes(of: handler) { $0.map { $0 } }
        subscribe(with: identifier, handler: handler)
        return identifier
    }
    
    @inlinable @discardableResult
    public func addHandler(on dispatcher: Dispatcher, handler: @escaping OutputHandler) -> HandlerIdentifier {
        addHandler { result in dispatcher.dispatchBlock { handler(result) } }
    }
    
    @inlinable @discardableResult
    public func addErrorHandler(on dispatcher: Dispatcher = .sync, execute handler: @escaping (Error) -> Void) -> HandlerIdentifier {
        addHandler(on: dispatcher) { result in
            if case .failure(let error) = result { handler(error) }
        }
    }
    
}
