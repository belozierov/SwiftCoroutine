//
//  XCTOrderedExpectation.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 11.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest

class XCTOrderedExpectation {
    
    fileprivate let expectations: [XCTestExpectation]
    
    init(description: String = "Ordered expectation", count: Int, file: StaticString = #file, line: Int = #line) {

        #if os(macOS)
        expectations = (0..<count)
            .map { "\(description) \($0)" }
            .map(XCTestExpectation.init(description:))
        #else
        expectations = (0..<count)
            .map { ("\(description) \($0)", file, line) }
            .map(XCTestExpectation.init(description:file:line:))
        #endif
    }
    
    init(expectations: [XCTestExpectation]) {
        self.expectations = expectations
    }
    
    func fulfill(_ index: Int) {
        expectations[index].fulfill()
    }
    
}

extension XCTestCase {
    
    func wait(for expectation: XCTOrderedExpectation, timeout: TimeInterval) {
        wait(for: expectation.expectations, timeout: timeout, enforceOrder: true)
    }
    
}
