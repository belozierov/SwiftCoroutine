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
    
    func subscribe(with identifier: AnyHashable, handler: @escaping OutputHandler)
    func unsubscribe(_ identifier: AnyHashable) -> OutputHandler?
    func await() throws -> Output
    
}

extension CoPublisher {
    
    @inlinable @discardableResult
    public func addHandler(_ handler: @escaping OutputHandler) -> AnyHashable {
        let identifier = withUnsafeBytes(of: handler) { $0.map { $0 } }
        subscribe(with: identifier, handler: handler)
        return identifier
    }
    
    @inlinable
    public func subscribe(with identifier: AnyHashable, handler: @escaping () -> Void) {
        subscribe(with: identifier) { _ in handler() }
    }
    
}





